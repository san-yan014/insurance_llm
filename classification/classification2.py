"""
Two modes:

Single-file batch mode (same as before, now via Message Batches API):
    python classification.py --input batches/15_batch_4.json

Sampling-loop mode (for Standard): keeps drawing random articles from --pool
until --target of them have passed the relevance check, logging irrelevant
ones to --excluded-log so future runs never re-spend tokens on them.
    python classification.py --pool standard_combined.json --target 150
"""
import anthropic
import json
import time
import textwrap
import os
import random
import argparse
from pathlib import Path
import pandas as pd

CLASSIFY_MODEL = "claude-sonnet-4-6"
CHECK_MODEL = "claude-haiku-4-5-20251001"

# same safety margins as batch_pipeline.py's extraction stages, so both scripts
# hit the same wall the same way instead of drifting apart over time
MAX_BATCH_BYTES = 200_000_000
MAX_BATCH_REQUESTS = 10_000
POLL_INTERVAL_SECONDS = 30

def request_size(req):
    # these are plain-text requests, not batch_pipeline.py's PDF documents,
    # so size is just the serialized request itself
    return len(json.dumps(req["params"]))

def chunk_requests(requests, max_bytes=MAX_BATCH_BYTES, max_count=MAX_BATCH_REQUESTS):
    chunks, current, current_size = [], [], 0
    for req in requests:
        size = request_size(req)
        if current and (current_size + size > max_bytes or len(current) >= max_count):
            chunks.append(current)
            current, current_size = [], 0
        current.append(req)
        current_size += size
    if current:
        chunks.append(current)
    return chunks

def submit_batches(client, requests, label):
    if not requests:
        return []
    chunks = chunk_requests(requests)
    batch_ids = []
    for i, chunk in enumerate(chunks):
        batch = client.messages.batches.create(requests=chunk)
        batch_ids.append(batch.id)
        print(f"{label}: submitted batch {i + 1}/{len(chunks)} ({len(chunk)} requests) -> {batch.id}")
    return batch_ids

def wait_for_batches(client, batch_ids, label):
    pending = set(batch_ids)
    while pending:
        for bid in list(pending):
            batch = client.messages.batches.retrieve(bid)
            if batch.processing_status == "ended":
                pending.discard(bid)
                counts = getattr(batch, "request_counts", None)
                ok = getattr(counts, "succeeded", "?")
                err = getattr(counts, "errored", "?")
                print(f"{label}: batch {bid} ended ({ok} succeeded, {err} errored)")
        if pending:
            time.sleep(POLL_INTERVAL_SECONDS)
    print(f"{label}: all {len(batch_ids)} batch(es) ended")

def collect_results(client, batch_ids):
    for bid in batch_ids:
        for entry in client.messages.batches.results(bid):
            yield entry.custom_id, entry.result

def message_text(message):
    return "".join(b.text for b in message.content if b.type == "text")

SYSTEM_PROMPT = """You are analyzing articles from Rough Notes, a trade publication for independent insurance agents (founded 1878).

This research examines how the insurance industry narrates change over time — how agents, agencies, and the broader industry respond to and make sense of external pressures. This includes how market forces, economic shocks, natural disasters, technology, regulation, and consolidation shape the way agents talk about their work, their clients, their businesses, and their identity as professionals.

Only extract passages that substantively engage with these dynamics. Ignore surface mentions, generic product descriptions, or content with no meaningful connection to industry pressures, agent behavior, or structural changes in the insurance landscape.

For each article, identify all relevant themes present and extract the specific passage that best represents each theme.

FRAMING - whose interests are centered in the passage:
- client-oriented: the passage centers the client's needs, protection, or wellbeing as the primary concern
- agent-oriented: the passage centers the agent's business interests, revenue, or self-interest as the primary concern
- mixed: both are genuinely and equally present — use sparingly

TOPICS - identify all themes present in the article and extract a passage of at least 75 words for each.

The (positive), (negative) and (neutral) subcategories apply ONLY to agent-client relationship and no other topic.

- agent-client relationship: the dynamic and nature of the relationship between agents and their clients. Focus on passages where the relationship itself — its quality, nature, tone, or power dynamic — is what's being discussed. Determine whether the dynamic is positive, negative, or neutral and append it to the topic label accordingly. Do not use this category when friction or tension originates from external factors like carrier pricing, market conditions, or systemic industry problems — use market conditions instead.
  - (positive): the dynamic is explicitly warm, trust-based, collaborative, or informal. There must be clear language indicating the relationship is genuinely valued beyond routine professional interaction.
  - (negative): the dynamic is explicitly tense, adversarial, or conflicted. There must be clear evidence of friction, broken trust, competing interests, or client harm that originates from the agent-client dynamic itself — not from external market or carrier factors.
  - (neutral): the interaction is professional and routine — agents doing their job without evident warmth or tension. Includes transactional exchanges, standard service descriptions, and procedural client interactions.

- market conditions: external forces worsening insurance availability or affordability. Includes hard markets, rising premiums, non-renewals, carriers exiting markets, insurer of last resort, FAIR Plan, Coastal Plan, clients being dropped or deemed uninsurable, CAT-driven market deterioration, and structural pricing or process problems that create friction between agents, carriers, and clients.

- technology/automation: technology directly changing or threatening the agent's role and relationship with clients. Includes direct-to-consumer platforms bypassing agents, algorithmic underwriting removing agent judgment, insurtech disrupting traditional distribution, and automation changing how agents interact with clients. Exclude generic mentions of technology tools with no connection to these dynamics.

- consolidation: ownership change and agency consolidation. Includes acquisitions by larger firms or private equity, perpetuation and succession planning, MGA mergers and acquisitions, retirement without succession forcing sales, agency valuation, and the trend of independent agencies being absorbed into corporate structures.

- family/community: independent agencies rooted in family ownership, community identity, or personal relationships. Includes generational succession, local agent identity, and the small business ethos that differentiates independent agents from corporate brokers.

- legal/regulatory: legislation, court rulings, or regulatory changes that meaningfully shift the power dynamic between agents, carriers, and clients. Includes fiduciary duty debates, state rate regulation, disclosure mandates, and laws that change what agents can or must do for clients. Exclude routine compliance mentions, technical coverage disputes, or minor court cases with no broader industry implications.

Respond in JSON format only as a list of entries, one per theme identified:
[
    {
        "topic": "topic name",
        "framing": "client-oriented | agent-oriented | mixed",
        "relevant_text": "a passage of at least 75 words that substantively discusses this theme",
        "reason": "one sentence explaining the framing and what tension or argument the passage captures",
        "other_theme": "brief phrase if the article contains a theme not listed above, otherwise null"
    }
]"""

EDUCATION_TOPICS = {
    "agent-client relationship (positive)",
    "agent-client relationship (negative)",
    "agent-client relationship (neutral)",
    "market conditions",
    "market conditions (negative relationship)",
    "market conditions (positive relationship)",
    "legal/regulatory",
}

def normalize_article(article):
    return {
        "title": article.get("title") or article.get("article_title", ""),
        "url": article.get("url", ""),
        "date": article.get("date", ""),
        "source": article.get("source") or article.get("publication", ""),
        "matched_keywords": article.get("matched_keywords", []),
        "text": article.get("text", ""),
    }

# stable key for tracking which articles have already been tried, across runs
def article_key(article):
    fname = article.get("filename", "")
    title = article.get("title") or article.get("article_title", "")
    return f"{fname}|{title}" if fname else title

def build_user_prompt(article):
    lines = []
    if article["matched_keywords"]:
        lines.append(f"Matched keywords: {', '.join(article['matched_keywords'])}")
    lines.append(f"Article title: {article['title']}")
    lines.append(f"Article text: {article['text']}")
    return "\n\n".join(lines)

def relevance_prompt(text):
    return """Does this article contain an agent or agency as a named or quoted participant AND describe some form of change, pressure, or adaptation in how they work with clients, carriers, or the broader market? Answer yes or no only.

Article text: """ + text[:3000]

def client_education_prompt(entry):
    return """The following are gold standard examples of client education:

Example 1: "We may discover that someone hasn't purchased a certain kind of coverage thinking it's not needed or being afraid that the premium is too high. That gives us the opportunity to explain how the coverage works and why it makes sense to add that coverage to the insurance program."

Example 2: "As the people responsible for supplying clients with the best information and advice on how to protect their business, agents and brokers need to communicate the benefits of supplemental coverages to protect clients from events they may have overlooked or assumed their existing policies would cover."

Does this passage show something similar — where an agent is explaining coverage, correcting a client misconception, or proactively informing a client about gaps or risks they were unaware of? Answer yes or no only.

Passage: """ + entry["relevant_text"]

def negative_relationship_prompt(entry):
    return """Does this passage explicitly show friction, tension, or dissatisfaction in the relationship between an agent and a client — such as a client being upset, an agent being blamed, or a breakdown in trust between them? Answer yes or no only.

Passage: """ + entry["relevant_text"]

def positive_relationship_prompt(entry):
    return """Does this passage explicitly show a favorable market moment that benefits clients — such as a soft market where premiums are falling, coverage is more accessible, clients have more negotiating power, or agents describe increased options or improved terms for clients? Answer yes or no only.

Passage: """ + entry["relevant_text"]

# builds one batch request per item, plus a lookup back to the original item (avoids encoding metadata into custom_id)
def make_batch_requests(items, build_params):
    requests, lookup = [], {}
    for i, item in enumerate(items):
        cid = f"req_{i}"
        requests.append({"custom_id": cid, "params": build_params(item)})
        lookup[cid] = item
    return requests, lookup

def run_batch(client, requests, lookup, label):
    batch_ids = submit_batches(client, requests, label)
    wait_for_batches(client, batch_ids, label)

    results = {}
    for custom_id, result in collect_results(client, batch_ids):
        item = lookup[custom_id]
        if result.type == "succeeded":
            results[custom_id] = (item, message_text(result.message))
        else:
            print(f"  [{label}] batch item failed: {custom_id} ({result.type})")
            results[custom_id] = (item, None)
    return results

def run_relevance_batch(client, articles):
    def params(article):
        return {
            "model": CHECK_MODEL, "max_tokens": 10, "temperature": 0,
            "messages": [{"role": "user", "content": relevance_prompt(article.get("text", ""))}],
        }
    requests, lookup = make_batch_requests(articles, params)
    results = run_batch(client, requests, lookup, "relevance")

    relevant, irrelevant = [], []
    for article, text in results.values():
        yes = bool(text) and text.strip().lower().startswith("yes")
        (relevant if yes else irrelevant).append(article)
    return relevant, irrelevant

def run_classification_batch(client, articles):
    def params(article):
        norm = normalize_article(article)
        return {
            "model": CLASSIFY_MODEL, "max_tokens": 4000, "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": build_user_prompt(norm)}],
        }
    requests, lookup = make_batch_requests(articles, params)
    results = run_batch(client, requests, lookup, "classify")

    parsed = {}
    for article, text in results.values():
        key = article_key(article)
        if not text:
            parsed[key] = (article, [])
            continue
        raw = text.strip().replace("```json", "").replace("```", "").strip()
        try:
            entries = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  JSON error for {article_key(article)}: {e}")
            entries = []
        parsed[key] = (article, entries)
    return parsed

# entries_with_articles: list of (article, entry) tuples, mutated in place
def run_check_batch(client, entries_with_articles, prompt_fn, label):
    def params(pair):
        _, entry = pair
        return {
            "model": CHECK_MODEL, "max_tokens": 10, "temperature": 0,
            "messages": [{"role": "user", "content": prompt_fn(entry)}],
        }
    requests, lookup = make_batch_requests(entries_with_articles, params)
    results = run_batch(client, requests, lookup, label)

    yes_ids = set()
    for pair, text in results.values():
        if text and text.strip().lower().startswith("yes"):
            yes_ids.add(id(pair[1]))
    return yes_ids

def apply_negative_check(client, entries_with_articles):
    targets = [p for p in entries_with_articles if p[1]["topic"] == "market conditions"]
    if not targets:
        return
    yes_ids = run_check_batch(client, targets, negative_relationship_prompt, "negative-relationship")
    for _, entry in targets:
        if id(entry) in yes_ids:
            entry["topic"] = "market conditions (negative relationship)"

def apply_positive_check(client, entries_with_articles):
    targets = [p for p in entries_with_articles if p[1]["topic"] == "market conditions"]
    if not targets:
        return
    yes_ids = run_check_batch(client, targets, positive_relationship_prompt, "positive-relationship")
    for _, entry in targets:
        if id(entry) in yes_ids:
            entry["topic"] = "market conditions (positive relationship)"

def apply_education_check(client, entries_with_articles):
    targets = [p for p in entries_with_articles if p[1]["topic"] in EDUCATION_TOPICS]
    if not targets:
        return
    yes_ids = run_check_batch(client, targets, client_education_prompt, "client-education")
    for _, entry in targets:
        if id(entry) not in yes_ids:
            continue
        topic = entry["topic"]
        if topic == "market conditions":
            entry["topic"] = "market conditions (client education)"
        elif topic == "market conditions (negative relationship)":
            entry["topic"] = "market conditions (negative relationship + client education)"
        elif topic == "market conditions (positive relationship)":
            entry["topic"] = "market conditions (positive relationship + client education)"
        elif topic == "legal/regulatory":
            entry["topic"] = "legal/regulatory (client education)"
        else:
            entry["topic"] = topic.replace(")", " + client education)")

# the model occasionally names the rationale field 'framing_reason' instead of the
# schema's 'reason' -- same content, different key, so alias it rather than discard it
def normalize_entry(entry):
    if "reason" not in entry and "framing_reason" in entry:
        entry["reason"] = entry["framing_reason"]
    return entry

# runs relevance -> classify -> refinement checks for one chunk of articles
def process_round(client, articles):
    relevant, irrelevant = run_relevance_batch(client, articles)
    if not relevant:
        return [], irrelevant, relevant

    parsed = run_classification_batch(client, relevant)

    entries_with_articles = []
    for article, entries in parsed.values():
        for entry in entries:
            entry = normalize_entry(entry)
            if all(entry.get(k) for k in ("topic", "relevant_text", "framing", "reason")):
                entries_with_articles.append((article, entry))
            else:
                print(f"  WARNING: skipping malformed entry from {article_key(article)}: {entry}")

    apply_negative_check(client, entries_with_articles)
    apply_positive_check(client, entries_with_articles)
    apply_education_check(client, entries_with_articles)

    rows = []
    for article, entry in entries_with_articles:
        norm = normalize_article(article)
        rows.append({
            "title": norm["title"],
            "url": norm["url"],
            "date": norm["date"],
            "publication": norm["source"],
            "article_key": article_key(article),
            "topic": entry["topic"],
            "framing": entry["framing"],
            "relevant_text": entry["relevant_text"],
            "reason": entry["reason"],
            "other_theme": entry.get("other_theme", None),
        })

    return rows, irrelevant, relevant

def load_json_list(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else []

def save_json_list(path, items):
    Path(path).write_text(json.dumps(items, ensure_ascii=False, indent=2))

def load_existing_rows(path):
    if not Path(path).exists():
        return []
    return pd.read_excel(path).to_dict("records")

def run_sampling_loop(client, pool_path, output_path, excluded_path, relevant_log_path, target, chunk_size, seed):
    pool = json.loads(Path(pool_path).read_text())
    excluded = load_json_list(excluded_path)
    relevant_log = load_json_list(relevant_log_path)
    all_rows = load_existing_rows(output_path)

    tried_keys = {e["key"] for e in excluded} | {e["key"] for e in relevant_log}
    coded_so_far = len({row["article_key"] for row in all_rows})
    print(f"{coded_so_far} unique articles with classified content already in output, target {target}")

    rng = random.Random(seed)

    while coded_so_far < target:
        candidates = [a for a in pool if article_key(a) not in tried_keys]
        if not candidates:
            print("pool exhausted before reaching target")
            break

        chunk = rng.sample(candidates, min(chunk_size, len(candidates)))
        try:
            rows, irrelevant, relevant = process_round(client, chunk)
        except Exception as e:
            print(f"round FAILED (chunk not marked as tried, will be resampled later): {e}")
            continue

        for article in irrelevant:
            key = article_key(article)
            norm = normalize_article(article)
            excluded.append({"key": key, "title": norm["title"], "publication": norm["source"], "reason": "irrelevant"})
            tried_keys.add(key)

        for article in relevant:
            key = article_key(article)
            norm = normalize_article(article)
            relevant_log.append({"key": key, "title": norm["title"], "publication": norm["source"]})
            tried_keys.add(key)

        all_rows.extend(rows)
        coded_so_far = len({row["article_key"] for row in all_rows})

        save_json_list(excluded_path, excluded)
        save_json_list(relevant_log_path, relevant_log)
        pd.DataFrame(all_rows).to_excel(output_path, index=False)

        new_unique = len({r["article_key"] for r in rows})
        print(f"round done: +{len(relevant)} relevant, +{len(irrelevant)} irrelevant, +{new_unique} unique articles coded -> {coded_so_far}/{target} unique articles coded")

    print(f"finished: {coded_so_far} unique articles coded, {len(relevant_log)} passed relevance in total, {len(excluded)} excluded total, {len(all_rows)} entries in output")

def run_single_file(client, input_path, output_path):
    articles = json.loads(Path(input_path).read_text())
    rows, irrelevant, relevant = process_round(client, articles)

    pd.DataFrame(rows).to_excel(output_path, index=False)
    print(f"classified {len(relevant)} relevant of {len(articles)} articles, {len(rows)} entries -> {output_path}")

    if irrelevant:
        excluded_path = os.path.splitext(output_path)[0] + "_excluded.json"
        save_json_list(excluded_path, [{"key": article_key(a), "title": normalize_article(a)["title"], "publication": normalize_article(a)["source"], "reason": "irrelevant"} for a in irrelevant])
        print(f"{len(irrelevant)} irrelevant articles logged -> {excluded_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="fixed batch file to classify once (single-file mode)")
    parser.add_argument("--pool", help="full article pool to sample from (sampling-loop mode)")
    parser.add_argument("--target", type=int, default=150, help="number of relevant articles to reach")
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="output xlsx filename")
    parser.add_argument("--excluded-log", default="excluded_articles.json")
    parser.add_argument("--relevant-log", default="relevant_articles.json")
    args = parser.parse_args()

    client = anthropic.Anthropic()

    if args.pool:
        output = args.output or "standard_classified.xlsx"
        run_sampling_loop(client, args.pool, output, args.excluded_log, args.relevant_log, args.target, args.chunk_size, args.seed)
    elif args.input:
        batch_name = os.path.basename(args.input).replace(".json", "")
        output = args.output or f"{batch_name}_classified.xlsx"
        run_single_file(client, args.input, output)
    else:
        parser.error("specify either --input (single-file mode) or --pool (sampling-loop mode)")

if __name__ == "__main__":
    main()