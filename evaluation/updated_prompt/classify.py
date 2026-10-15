"""
This script has updated prompt definitions.
    - Removed positive/negative for market conditions
    - Removed agent-client framings 
    - Removed other_theme fallback 
    - Added a disambiguation rule for market condtions, legal/regulatory, and technology/automation codes since these are the main categories that the LLM failed to differentiate clearly + a few examples for LLM to make decisions easier 

    Everything else remains the same 


Input and Output: 

    Same format - output will have columns starting from llm_code filled out.  

    coder	article_key	title	relevant_text	our_code	llm_code	
    
    llm_reasoning	match


    Script will also print out:
        classifying 258 rows
        10/258 done
        20/258 done
        ...
        250/258 done

        edna: 44 matches out of 72 (61.1%), F1 score = 0.XXX
        True: 44, False: 28
        helen: 55 matches out of 88 (62.5%), F1 score = 0.XXX
        True: 55, False: 33
        will: 78 matches out of 98 (79.6%), F1 score = 0.XXX
        True: 78, False: 20

        combined: 177 matches out of 258 (68.6%), F1 score = 0.XXX
        True: 177, False: 81

        saved to new_prompt_comparison_template_classified.xlsx


Script usage: python classify_passages.py <workbook.xlsx> [output.xlsx]
"""
import sys
import json
import time
import pandas as pd
from anthropic import Anthropic

SHEET_NAME = "llm vs handcoded texts"
MODEL = "claude-sonnet-4-6"
CHECK_MODEL = "claude-haiku-4-5-20251001"

VALID_CODES = {
    "agent-client relationship (positive)",
    "agent-client relationship (negative)",
    "agent-client relationship (neutral)",
    "market conditions",
    "technology/automation",
    "consolidation",
    "family/community",
    "legal/regulatory",
}

EDUCATION_CODES = {
    "agent-client relationship (positive)",
    "agent-client relationship (negative)",
    "agent-client relationship (neutral)",
    "market conditions",
    "legal/regulatory",
}

CATEGORY_DEFINITIONS = """You are analyzing passages from Rough Notes, a trade publication for independent insurance agents (founded 1878).

This research examines how the insurance industry narrates change over time — how agents, agencies, and the broader industry respond to and make sense of external pressures. This includes how market forces, economic shocks, natural disasters, technology, regulation, and consolidation shape the way agents talk about their work, their clients, their businesses, and their identity as professionals.

You will be given a single passage that has already been extracted from an article. For each passage, classify it into exactly ONE of the following categories.

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

You must always choose the single best-fitting category from the list above, even if the fit is imperfect. Never use a label outside this list.

Respond with ONLY a JSON object, no other text, in this exact format:
{"code": "<category label exactly as defined above>", "reasoning": "<one or two sentence explanation>"}"""


def base_code(label):
    if pd.isna(label):
        return None
    return label.split(" (")[0].strip()


def is_valid_code(code):
    if code in VALID_CODES:
        return True
    if code.startswith("agent-client relationship ("):
        inner = code[len("agent-client relationship ("):-1]
        sentiment = inner.replace(" + client education", "").strip()
        return f"agent-client relationship ({sentiment})" in VALID_CODES
    stripped = code.replace(" (client education)", "").strip()
    return stripped in VALID_CODES


def classify(client, text):
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        temperature=0,
        messages=[{
            "role": "user",
            "content": f"{CATEGORY_DEFINITIONS}\n\nPassage to classify:\n\n{text}",
        }],
    )
    raw = response.content[0].text.strip()
    parsed = json.loads(raw)
    return parsed["code"], parsed["reasoning"]


def client_education_prompt(text):
    return """The following are gold standard examples of client education:

Example 1: "We may discover that someone hasn't purchased a certain kind of coverage thinking it's not needed or being afraid that the premium is too high. That gives us the opportunity to explain how the coverage works and why it makes sense to add that coverage to the insurance program."

Example 2: "As the people responsible for supplying clients with the best information and advice on how to protect their business, agents and brokers need to communicate the benefits of supplemental coverages to protect clients from events they may have overlooked or assumed their existing policies would cover."

Does this passage show something similar — where an agent is explaining coverage, correcting a client misconception, or proactively informing a client about gaps or risks they were unaware of? Answer yes or no only.

Passage: """ + text


def check_client_education(client, text):
    response = client.messages.create(
        model=CHECK_MODEL,
        max_tokens=10,
        temperature=0,
        messages=[{"role": "user", "content": client_education_prompt(text)}],
    )
    answer = response.content[0].text.strip().lower()
    return answer.startswith("yes")


def apply_education_suffix(code):
    if code == "market conditions":
        return "market conditions (client education)"
    if code == "legal/regulatory":
        return "legal/regulatory (client education)"
    return code.replace(")", " + client education)")


def match_key(label):
    if pd.isna(label):
        return None
    base = label.split(" (")[0].strip()
    if base == "market conditions":
        return base
    return label


def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def weighted_f1(y_true, y_pred):
    labels = set(y_true.dropna()) | set(y_pred.dropna())
    weighted_sum = 0.0
    weight_total = 0
    for label in labels:
        tp = ((y_true == label) & (y_pred == label)).sum()
        fp = ((y_true != label) & (y_pred == label)).sum()
        fn = ((y_true == label) & (y_pred != label)).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        weight = (y_true == label).sum()
        weighted_sum += f1 * weight
        weight_total += weight
    return weighted_sum / weight_total if weight_total else 0.0


def report(name, sub):
    acc = accuracy(sub["our_code"].map(match_key), sub["llm_code"].map(match_key))
    f1 = weighted_f1(sub["our_code"].map(match_key), sub["llm_code"].map(match_key))
    n_match = sub["match"].sum()
    n_mismatch = (~sub["match"]).sum()
    print(f"{name}: {n_match} matches out of {len(sub)} ({acc:.1%}), F1 score = {f1:.3f}")
    print(f"  True: {n_match}, False: {n_mismatch}")


def main():
    if len(sys.argv) < 2:
        print("usage: python classify_passages.py <workbook.xlsx> [output.xlsx]")
        sys.exit(1)

    workbook = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else workbook.rsplit(".", 1)[0] + "_classified.xlsx"

    df = pd.read_excel(workbook, sheet_name=SHEET_NAME)
    df["llm_code"] = df["llm_code"].astype(object)
    df["llm_reasoning"] = df["llm_reasoning"].astype(object)
    client = Anthropic()

    to_classify = df[df["llm_code"].isna() & df["our_code"].notna()]
    print(f"classifying {len(to_classify)} rows")

    for i, idx in enumerate(to_classify.index, start=1):
        text = df.at[idx, "relevant_text"]
        try:
            code, reasoning = classify(client, text)
            if code in EDUCATION_CODES and check_client_education(client, text):
                code = apply_education_suffix(code)
        except Exception as e:
            print(f"  row {idx} failed: {e}")
            continue
        if not is_valid_code(code):
            print(f"  WARNING: row {idx} got unrecognized code '{code}'")
        df.at[idx, "llm_code"] = code
        df.at[idx, "llm_reasoning"] = reasoning
        if i % 10 == 0:
            print(f"  {i}/{len(to_classify)} done")
        time.sleep(0.5)

    df["match"] = df["our_code"].map(match_key) == df["llm_code"].map(match_key)

    print()
    for coder in sorted(df["coder"].dropna().unique()):
        report(coder, df[df["coder"] == coder])
    report("combined", df)
    print()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=SHEET_NAME, index=False)

    print(f"saved to {output}")


if __name__ == "__main__":
    main()