import base64
import json
import sys
import time
from pathlib import Path

from pypdf import PdfReader

from pipeline import (
    IDENTIFY_PROMPT, EXTRACT_PROMPT, MODEL, PUBLICATION,
    parse_filename, extract_json, build_subset_pdf, ends_cleanly,
)

MAX_BATCH_BYTES = 200_000_000   # safety margin under Anthropic's 256MB per-batch payload limit
MAX_BATCH_REQUESTS = 10_000     # safety margin under the 100,000-requests-per-batch limit
MAX_EXTRACT_ROUNDS = 4          # matches the synchronous pipeline's page-expansion cap
MAX_IDENTIFY_ROUNDS = 3         # retry rounds for transient failures/parse errors, not page expansion
POLL_INTERVAL_SECONDS = 30

def is_valid_article_entry(a):
    return (
        isinstance(a, dict)
        and "article_title" in a
        and isinstance(a.get("pages"), list)
        and bool(a["pages"])
    )

def request_size(req):
    # dominated by the base64 document payload -- a cheap but safe estimate for chunking
    total = 0
    for block in req["params"]["messages"][0]["content"]:
        if isinstance(block, dict) and block.get("type") == "document":
            total += len(block["source"]["data"])
    return total or 2000

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
    """Submit requests as one or more size-bounded batches. Returns list of batch ids."""
    if not requests:
        return []
    chunks = chunk_requests(requests)
    batch_ids = []
    for i, chunk in enumerate(chunks):
        batch = client.messages.batches.create(requests=chunk)
        batch_ids.append(batch.id)
        print(f"{label}: submitted batch {i + 1}/{len(chunks)} ({len(chunk)} requests) -> {batch.id}")
    return batch_ids

HEARTBEAT_EVERY_N_POLLS = 10  # ~5 minutes at the 30s poll interval

def wait_for_batches(client, batch_ids, label):
    pending = set(batch_ids)
    start = time.time()
    poll_count = 0
    while pending:
        still_waiting = []
        for bid in list(pending):
            batch = client.messages.batches.retrieve(bid)
            counts = getattr(batch, "request_counts", None)
            if batch.processing_status == "ended":
                pending.discard(bid)
                ok = getattr(counts, "succeeded", "?")
                err = getattr(counts, "errored", "?")
                print(f"{label}: batch {bid} ended ({ok} succeeded, {err} errored)")
            else:
                proc = getattr(counts, "processing", "?")
                ok = getattr(counts, "succeeded", "?")
                err = getattr(counts, "errored", "?")
                still_waiting.append(f"{bid}: processing={proc} succeeded={ok} errored={err}")
        if pending:
            poll_count += 1
            if poll_count % HEARTBEAT_EVERY_N_POLLS == 0:
                elapsed_min = round((time.time() - start) / 60, 1)
                print(f"{label}: still waiting after {elapsed_min}m -- " + "; ".join(still_waiting))
            time.sleep(POLL_INTERVAL_SECONDS)
    print(f"{label}: all {len(batch_ids)} batch(es) ended")

def collect_results(client, batch_ids):
    """Yields (custom_id, result) for every request across the given batches."""
    for bid in batch_ids:
        for entry in client.messages.batches.results(bid):
            yield entry.custom_id, entry.result

def document_request(custom_id, model, max_tokens, pdf_bytes, prompt):
    pdf_b64 = base64.b64encode(pdf_bytes).decode()
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        },
    }

def message_text(message):
    return "".join(b.text for b in message.content if b.type == "text")

# these error types are deterministic given the same input -- retrying wastes a full round-trip
# since the result will be identical every time (unlike overloaded_error/rate_limit_error/
# api_error/timeout, which are genuinely worth retrying)
PERMANENT_ERROR_TYPES = {"invalid_request_error", "authentication_error", "permission_error", "not_found_error"}

def is_permanent_error(result):
    err = getattr(result, "error", None)
    inner = getattr(err, "error", None) if err is not None else None
    etype = getattr(inner, "type", None) if inner is not None else None
    return etype in PERMANENT_ERROR_TYPES

def describe_error(result):
    """Pulls the real error detail out of an errored batch result (Anthropic's error envelope
    is doubly-nested: result.error.error.{type,message}). Falls back gracefully if the shape
    doesn't match what's expected, since error-logging itself should never crash the pipeline."""
    err = getattr(result, "error", None)
    if err is None:
        return getattr(result, "type", "unknown error")
    inner = getattr(err, "error", None)
    if inner is not None:
        etype = getattr(inner, "type", "unknown")
        emsg = getattr(inner, "message", "")
        return f"{etype}: {emsg}" if emsg else etype
    return str(err)

# ---- identify stage ----

MAX_PDF_BYTES = 33_554_432  # 32 MB -- Anthropic's hard limit on a single document's size

def run_identify_stage(client, pdf_paths, model, publication, state_dir):
    prompt = IDENTIFY_PROMPT.format(publication=publication)
    manifests = {}
    remaining = {}
    last_error = {}
    permanently_failed = {}

    for p in pdf_paths:
        p = Path(p)
        stem = p.stem
        size = p.stat().st_size
        if size > MAX_PDF_BYTES:
            detail = f"PDF too large to submit: {size} bytes > {MAX_PDF_BYTES} bytes (32 MB limit)"
            print(f"  identify SKIPPED for {stem} (oversized, never submitted): {detail}")
            permanently_failed[stem] = detail
            last_error[stem] = detail
        else:
            remaining[stem] = p

    for round_num in range(1, MAX_IDENTIFY_ROUNDS + 1):
        if not remaining:
            break
        id_map = {}  # custom_id -> stem, since custom_id can't contain ':' or other punctuation
        requests = []
        for i, (stem, pdf_path) in enumerate(remaining.items()):
            custom_id = f"identify-{i}"
            id_map[custom_id] = stem
            requests.append(document_request(custom_id, model, 4000, pdf_path.read_bytes(), prompt))

        batch_ids = submit_batches(client, requests, f"identify round {round_num}")
        wait_for_batches(client, batch_ids, f"identify round {round_num}")

        next_remaining = {}
        for custom_id, result in collect_results(client, batch_ids):
            stem = id_map[custom_id]
            if result.type != "succeeded":
                detail = describe_error(result)
                last_error[stem] = detail
                if is_permanent_error(result):
                    print(f"  identify FAILED for {stem} (permanent, not retrying): {detail}")
                    permanently_failed[stem] = detail  # still needs to be visible in the output, just not retried
                    continue
                print(f"  identify FAILED for {stem}: {detail}")
                next_remaining[stem] = remaining[stem]
                continue
            try:
                parsed = extract_json(message_text(result.message))
                cleaned = [a for a in parsed if is_valid_article_entry(a)]
                if not cleaned:
                    raise ValueError("no well-formed articles in response")
            except Exception as e:
                last_error[stem] = f"parse error: {e}"
                print(f"  identify PARSE FAILED for {stem}: {e}")
                next_remaining[stem] = remaining[stem]
                continue
            manifest = [{"article_title": a["article_title"], "pages": a["pages"]} for a in cleaned]
            manifests[stem] = manifest
            (state_dir / f"{stem}.manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        remaining = next_remaining

    failed = dict(permanently_failed)
    for stem in remaining:
        detail = last_error.get(stem, "unknown error")
        print(f"  identify exhausted {MAX_IDENTIFY_ROUNDS} retry rounds for {stem}: {detail}")
        failed[stem] = detail
    return manifests, failed

# ---- extract stage ----

EXTRACT_MAX_TOKENS_DEFAULT = 8000
EXTRACT_MAX_TOKENS_CAP = 32000

def run_extract_rounds(client, state, pdf_dir, model, publication, total_pages_cache):
    """Runs page-expansion rounds against whatever entries in `state` have done=False.
    run_extract_stage below builds that state fresh from a manifest for standard use."""
    for round_num in range(1, MAX_EXTRACT_ROUNDS + 1):
        pending = [k for k, s in state.items() if not s["done"] and not s["failed"]]
        if not pending:
            break

        id_map = {}
        requests = []
        for i, (stem, idx) in enumerate(pending):
            s = state[(stem, idx)]
            pdf_path = pdf_dir / f"{stem}.pdf"
            total_pages = total_pages_cache[stem]
            subset_bytes = build_subset_pdf(pdf_path, s["pages"], total_pages)
            prompt = EXTRACT_PROMPT.format(publication=publication, article_title=s["title"])
            custom_id = f"extract-{i}"
            id_map[custom_id] = (stem, idx)
            max_tokens = s.get("max_tokens", EXTRACT_MAX_TOKENS_DEFAULT)
            requests.append(document_request(custom_id, model, max_tokens, subset_bytes, prompt))

        batch_ids = submit_batches(client, requests, f"extract round {round_num}")
        wait_for_batches(client, batch_ids, f"extract round {round_num}")

        for custom_id, result in collect_results(client, batch_ids):
            stem, idx = id_map[custom_id]
            s = state[(stem, idx)]
            if result.type != "succeeded":
                detail = describe_error(result)
                print(f"  extract FAILED for {stem}#{idx}: {detail}")
                s["failed"] = True
                s["error"] = detail
                continue

            if result.message.stop_reason == "max_tokens":
                # the response got cut off mid-generation -- this is NOT a malformed response,
                # it's a token-budget problem, so it's retryable with more room, not a permanent
                # failure. Without this check the resulting unparseable JSON would otherwise look
                # just like any other parse error and get marked failed with no retry at all.
                old_budget = s.get("max_tokens", EXTRACT_MAX_TOKENS_DEFAULT)
                s["rounds"] += 1
                if old_budget < EXTRACT_MAX_TOKENS_CAP and s["rounds"] < MAX_EXTRACT_ROUNDS:
                    s["max_tokens"] = min(old_budget * 2, EXTRACT_MAX_TOKENS_CAP)
                    print(f"  extract TRUNCATED for {stem}#{idx}, retrying with max_tokens={s['max_tokens']}")
                else:
                    s["failed"] = True
                    s["error"] = f"response truncated even at max_tokens={old_budget}"
                    print(f"  extract FAILED for {stem}#{idx}: still truncated at the token cap")
                continue

            try:
                parsed = extract_json(message_text(result.message))
                if not isinstance(parsed, dict) or "text" not in parsed or not parsed["text"].strip():
                    raise ValueError("missing/empty text")
            except Exception as e:
                detail = f"parse error: {e}"
                print(f"  extract PARSE FAILED for {stem}#{idx}: {detail}")
                s["failed"] = True
                s["error"] = detail
                continue

            s["text"] = parsed["text"]
            s["rounds"] += 1
            if ends_cleanly(s["text"]):
                s["done"] = True
                continue
            total_pages = total_pages_cache[stem]
            next_page = max(s["pages"]) + 1
            if next_page > total_pages or next_page in s["pages"] or s["rounds"] >= MAX_EXTRACT_ROUNDS:
                s["done"] = True  # best effort -- flagged for review when writing output
            else:
                s["pages"].append(next_page)

    return state

def run_extract_stage(client, manifests, pdf_dir, model, publication, total_pages_cache):
    state = {}
    for stem, articles in manifests.items():
        for idx, art in enumerate(articles):
            state[(stem, idx)] = {
                "title": art["article_title"], "pages": list(art["pages"]),
                "text": None, "rounds": 0, "failed": False, "done": False, "error": None,
                "max_tokens": EXTRACT_MAX_TOKENS_DEFAULT,
            }
    return run_extract_rounds(client, state, pdf_dir, model, publication, total_pages_cache)

# ---- output ----

def write_outputs(state, out_dir, publication):
    by_stem = {}
    for (stem, idx), s in state.items():
        by_stem.setdefault(stem, []).append((idx, s))

    for stem, entries in by_stem.items():
        entries.sort(key=lambda pair: pair[0])
        date, issue = parse_filename(f"{stem}.pdf")
        results, errors, review = [], [], []
        for idx, s in entries:
            if s["failed"] or s["text"] is None:
                errors.append({"article_title": s["title"], "pages": s["pages"],
                                "error": s.get("error") or "failed after batch retries"})
                continue
            results.append({
                "filename": f"{stem}.pdf",
                "publication": publication,
                "date": date,
                "issue": issue,
                "article_title": s["title"],
                "text": " ".join(s["text"].split()),
            })
            if not ends_cleanly(s["text"]):
                review.append({"article_title": s["title"], "pages": s["pages"],
                                "reason": "still incomplete after max batch rounds"})
        (out_dir / f"{stem}_articles.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
        if errors:
            (out_dir / f"{stem}.errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False))
        if review:
            (out_dir / f"{stem}.review.json").write_text(json.dumps(review, indent=2, ensure_ascii=False))
        print(f"{stem}: {len(results)} articles, {len(errors)} failed, {len(review)} flagged for review")

def write_failed_articles_summary(out_dir):
    """Scans out_dir for individual articles that failed extraction while the rest of their PDF
    succeeded (a real article_title, not None -- see write_skip_summary for the PDF-level case).
    Separate file because these need a different fix: re-extracting or manually transcribing one
    article, not compressing/resubmitting a whole PDF. Same self-regenerating design as
    skipped_pdfs.txt -- rebuilt fresh from disk every time, so it stays correct across reruns."""
    out_dir = Path(out_dir)
    failed = []
    for errors_path in sorted(out_dir.glob("*.errors.json")):
        stem = errors_path.name[: -len(".errors.json")]
        try:
            errors = json.loads(errors_path.read_text())
        except Exception:
            continue
        for e in errors:
            if e.get("article_title") is not None:
                failed.append(f"{stem}.pdf | {e['article_title']} | pages {e.get('pages')} | "
                              f"{e.get('error', 'unknown error')}")

    summary_path = out_dir / "failed_articles.txt"
    if failed:
        summary_path.write_text("\n".join(failed) + "\n")
        print(f"{len(failed)} article(s) need manual handling -- see {summary_path.name}")
    elif summary_path.exists():
        summary_path.unlink()

def write_skip_summary(out_dir):
    """Scans out_dir for PDFs that never got past the identify stage at all (oversized, or
    permanently failed for any other reason) and writes one consolidated, human-readable list
    for manual follow-up. Regenerated fresh from the actual output folder every time this runs,
    so it stays correct across multiple reruns without depending on in-memory state."""
    out_dir = Path(out_dir)
    skipped = []
    for errors_path in sorted(out_dir.glob("*.errors.json")):
        stem = errors_path.name[: -len(".errors.json")]
        try:
            errors = json.loads(errors_path.read_text())
        except Exception:
            continue
        # a PDF-level failure (never got past identify) always has article_title=None;
        # a partial extract failure (some articles succeeded) always has a real title
        total_failures = [e for e in errors if e.get("article_title") is None]
        if total_failures:
            skipped.append(f"{stem}.pdf: {total_failures[0].get('error', 'unknown error')}")

    summary_path = out_dir / "skipped_pdfs.txt"
    if skipped:
        summary_path.write_text("\n".join(skipped) + "\n")
        print(f"{len(skipped)} PDF(s) need manual handling -- see {summary_path.name}")
    elif summary_path.exists():
        summary_path.unlink()  # nothing skipped anymore (e.g. fixed manually) -- clear the stale list

# ---- driver ----

def run_batch_pipeline(pdf_dir, out_dir, state_dir, model=MODEL, publication=PUBLICATION,
                        chunk_start=None, chunk_end=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        _run_batch_pipeline(pdf_dir, out_dir, state_dir, model, publication, chunk_start, chunk_end)
    finally:
        write_skip_summary(out_dir)  # always regenerate, even on an early return or a crash
        write_failed_articles_summary(out_dir)

def _run_batch_pipeline(pdf_dir, out_dir, state_dir, model, publication, chunk_start, chunk_end):
    pdf_dir, state_dir = Path(pdf_dir), Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    all_pdfs = sorted(pdf_dir.glob("*.pdf"))
    if chunk_start is not None or chunk_end is not None:
        chunk_start = chunk_start or 0
        chunk_end = chunk_end if chunk_end is not None else len(all_pdfs)
        chunk_pdfs = all_pdfs[chunk_start:chunk_end]
        print(f"chunk [{chunk_start}:{chunk_end}] of {len(all_pdfs)} total PDFs -> {len(chunk_pdfs)} in this chunk")
    else:
        chunk_pdfs = all_pdfs

    todo = [p for p in chunk_pdfs if not (out_dir / f"{p.stem}_articles.json").exists()]
    print(f"{len(chunk_pdfs)} PDFs in this run, {len(todo)} not yet done")
    if not todo:
        return

    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    total_pages_cache = {p.stem: len(PdfReader(str(p)).pages) for p in todo}

    manifests = {}
    need_identify = []
    for p in todo:
        mpath = state_dir / f"{p.stem}.manifest.json"
        if mpath.exists():
            manifests[p.stem] = json.loads(mpath.read_text())
        else:
            need_identify.append(p)

    if need_identify:
        new_manifests, failed_identify = run_identify_stage(client, need_identify, model, publication, state_dir)
        manifests.update(new_manifests)
        for stem, detail in failed_identify.items():
            (out_dir / f"{stem}.errors.json").write_text(json.dumps(
                [{"article_title": None, "pages": None, "error": f"identify stage failed: {detail}"}],
                indent=2, ensure_ascii=False))
            print(f"{stem}: identify failed permanently -- wrote {stem}.errors.json so this isn't silently missing")
        if failed_identify:
            # regenerate now, not just at the end -- if the job gets killed during the (often
            # much longer) extraction phase that follows, this failure is still reflected
            write_skip_summary(out_dir)

    if not manifests:
        print("no manifests produced -- nothing to extract")
        return

    state = run_extract_stage(client, manifests, pdf_dir, model, publication, total_pages_cache)
    write_outputs(state, out_dir, publication)
    write_failed_articles_summary(out_dir)

if __name__ == "__main__":
    if len(sys.argv) not in (4, 6):
        print("usage: python batch_pipeline.py <pdf_folder> <output_folder> <state_folder> [chunk_start chunk_end]")
        sys.exit(1)
    pdf_dir, out_dir, state_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    if len(sys.argv) == 6:
        run_batch_pipeline(pdf_dir, out_dir, state_dir, chunk_start=int(sys.argv[4]), chunk_end=int(sys.argv[5]))
    else:
        run_batch_pipeline(pdf_dir, out_dir, state_dir)