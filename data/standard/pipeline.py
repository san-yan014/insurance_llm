import base64
import json
import re
import sys
import time
from pathlib import Path

from pypdf import PdfReader, PdfWriter

MODEL = "claude-sonnet-5"
PUBLICATION = "The Standard"
MAX_RETRIES = 3

IDENTIFY_PROMPT = """This PDF is one issue of {publication}, a weekly insurance trade newspaper.
List every real news article in it. Do not extract article text yet -- just identify each article.

WHAT COUNTS AS AN ARTICLE (include):
- Any piece that opens with a wire-style dateline, e.g. "BOSTON — ..." or "WASHINGTON, D.C. — ...".
- A short item without an explicit dateline but written as straight news (a company action, regulatory
  filing, court ruling, personnel/M&A news reported as fact) -- these still count. Each headline marks a
  new, separate article, even if thematically related to the one immediately before it.
- A recurring signed analysis/opinion column written by a named outside professional with their title
  and firm (e.g. "By Jane Smith, Partner, XYZ LLP"), offering analysis rather than reporting news. Judge
  this by that byline-plus-analysis structure, not a specific section name. Do not confuse it with the
  publication's own in-house editor's column (unsigned, no outside byline) -- that is NOT an article.

WHAT TO EXCLUDE:
- Display and classified ads, even ones with large/bold headline-style text -- distinguish by content,
  not font: ads use promotional language and report no facts from an attributed source.
- The masthead/cover banner, "In This Issue" table of contents, Services Directory, Division of
  Insurance license notices, Calendar, Executive Suite (people-moves briefs), Sidelights, and the
  publication's own unsigned editor's column.

For each article, find every page it appears on -- including later pages where it resumes. Most
resuming pages are marked "continued on page N" / "continued from page N", but a long article can
also simply run past a page break with NO such label at all, as part of normal layout -- if an
article's text looks unfinished at the bottom of a page (cut off mid-sentence, no closing punctuation
or wrap-up), check whether a LATER page picks up that same subject in continuing prose. The resuming
page is not always the very next one: a full-page ad or an unrelated department (Calendar, Executive
Suite, etc.) can sit in between with no connection to the article at all -- keep scanning forward past
those irrelevant pages until you either find the resuming content or conclude the article doesn't
continue.

Output ONLY a JSON array (no other text, no markdown fences), one object per article:
[{{"article_title": "...", "pages": [1, 10]}}, ...]
"""

EXTRACT_PROMPT = """These pages are from {publication}. Extract ONLY this one article's text:

"{article_title}"

The pages may also contain other articles, ads, or page furniture sharing the same space -- ignore
everything that isn't this specific article.

HANDLING CONTINUATION ACROSS THESE PAGES:
- If this article breaks off on one page with "continued on page N" and resumes on a later page here
  with "continued from page N" (often with the headline repeated above the resuming text), splice the
  two parts into one continuous text -- the sentence cut off must connect grammatically with the
  sentence that resumes. Remove the "continued on/from page N" labels and any repeated
  headline/folio/page-number text from the final output -- that is page furniture, not content.
- An article can also continue onto a later page you were given with NO such label at all, simply as
  part of normal layout. If the text on one page doesn't reach a natural stopping point (cut off
  mid-sentence, no closing punctuation), check the following pages for continuing prose on the same
  subject -- the resuming page is not always the very next one; a full-page ad or unrelated department
  page can sit in between with no connection to this article at all. Skip past those and splice in the
  continuation wherever it actually resumes.

HANDLING SCRAMBLED READING ORDER:
- If this article shares a page with another article side by side in the same multi-column grid, the
  underlying text can extract in a scrambled order. Don't assume a simple top-to-bottom or left-to-right
  pass is correct if it produces text that doesn't read as coherent sentences -- reassemble by reading
  for grammatical and topical continuity, using only the chunks that belong to THIS article.

OTHER RULES:
- If this article has a byline (e.g. "By Jane Smith, Partner, XYZ LLP"), keep it as the first line of
  the extracted text. A byline is part of the article's content, not page furniture -- even when it sits
  directly next to a recurring section logo/header (e.g. a stylized column name like "Legal Dimensions")
  that you should otherwise treat as decoration. Keep the byline itself; drop only the decorative logo.
- Do not repeat the headline/article_title as the first sentence of the body text -- the title is
  already captured separately. Start the extracted text with the byline (if any) or the first actual
  sentence of reported content.
- Remove footnotes, endnotes, and citation numbers from the body text if this is a signed analysis column.
- If a pull-quote box repeats a sentence already quoted in the body, don't duplicate it.
- If the article ends with a small decorative symbol marking the end of the article (a filled or
  outline square, circle, or diamond -- e.g. "■"), drop that symbol; it is page furniture, not content.
- Preserve the original wording exactly (fix only line-wrap hyphenation artifacts, e.g. "car-\\nriers"
  -> "carriers", and normalize whitespace) -- do not paraphrase or summarize.

Output ONLY a JSON object (no other text, no markdown fences): {{"text": "..."}}
"""

def parse_filename(filename):
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})_(.+)\.pdf$", filename)
    if not m:
        return None, None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}", m.group(4)

def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start_chars, end_chars = "[{", "]}"
    start = next((i for i, c in enumerate(text) if c in start_chars), None)
    end = next((i for i, c in enumerate(reversed(text)) if c in end_chars), None)
    if start is None or end is None:
        raise ValueError(f"no JSON found in response: {text[:200]!r}")
    candidate = text[start: len(text) - end]
    return json.loads(candidate)

def call_claude(client, model, pdf_bytes, prompt, max_tokens):
    pdf_b64 = base64.b64encode(pdf_bytes).decode()
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    text = "".join(b.text for b in message.content if b.type == "text")
    return text, message.stop_reason

def call_with_retries(client, model, pdf_bytes, prompt, max_tokens, validate=None, max_retries=MAX_RETRIES):
    last_err = None
    for attempt in range(max_retries):
        try:
            text, stop_reason = call_claude(client, model, pdf_bytes, prompt, max_tokens)
            if stop_reason == "max_tokens":
                max_tokens = min(max_tokens * 2, 32000)
                last_err = RuntimeError(f"response truncated (max_tokens={max_tokens // 2}), retrying with {max_tokens}")
                time.sleep(1)
                continue
            result = extract_json(text)
            if validate is not None:
                validate(result)
            return result
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {max_retries} attempts: {last_err}")

def build_subset_pdf(pdf_path, pages, total_pages):
    valid_pages = sorted({p for p in pages if isinstance(p, int) and 1 <= p <= total_pages})
    if not valid_pages:
        raise ValueError(f"no valid page numbers in {pages} (pdf has {total_pages} pages)")
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    for p in valid_pages:
        writer.add_page(reader.pages[p - 1])
    from io import BytesIO
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()

def identify_articles(pdf_path, client, model, publication):
    pdf_bytes = Path(pdf_path).read_bytes()
    prompt = IDENTIFY_PROMPT.format(publication=publication)

    def validate(result):
        if not isinstance(result, list) or not result:
            raise ValueError(f"expected a non-empty JSON array, got {result!r}"[:200])
        cleaned = [a for a in result if _valid_article_entry(a)]
        if not cleaned:
            raise ValueError(f"none of the {len(result)} entries were well-formed")

    result = call_with_retries(client, model, pdf_bytes, prompt, max_tokens=4000, validate=validate)
    return [{"article_title": a["article_title"], "pages": a["pages"]} for a in result if _valid_article_entry(a)]

def _valid_article_entry(a):
    return (
        isinstance(a, dict)
        and "article_title" in a
        and isinstance(a.get("pages"), list)
        and bool(a["pages"])
    )

ENDMARKS = "\u25A0\u25A1\u25CF\u25CB\u25AA\u25AB\u25C6\u25C7"

def ends_cleanly(text):
    # the endmark is the publisher's own "article is done" signal -- trust it on its own,
    # since the last line before it can be a bullet point with no terminal punctuation at all
    text = text.strip()
    if text and text[-1] in ENDMARKS:
        return True
    return bool(re.search(r'[.!?]["\u201d\u2019\']?\s*$', text))

def extract_article_text(pdf_path, article, client, model, publication, total_pages):
    pages = list(article["pages"])
    text = None
    for _ in range(4):
        subset_bytes = build_subset_pdf(pdf_path, pages, total_pages)
        prompt = EXTRACT_PROMPT.format(publication=publication, article_title=article["article_title"])

        def validate(result):
            if not isinstance(result, dict) or "text" not in result or not result["text"].strip():
                raise ValueError(f"expected a non-empty 'text' key, got {result!r}"[:200])

        result = call_with_retries(client, model, subset_bytes, prompt, max_tokens=8000, validate=validate)
        text = result["text"]
        if ends_cleanly(text):
            return text, False
        next_page = max(pages) + 1
        if next_page > total_pages or next_page in pages:
            break
        pages.append(next_page)
    return text, True

def process_pdf(pdf_path, out_dir, client, model=MODEL, publication=PUBLICATION):
    pdf_path, out_dir = Path(pdf_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem
    final_path = out_dir / f"{stem}_articles.json"
    manifest_path = out_dir / f"{stem}.manifest.json"
    errors_path = out_dir / f"{stem}.errors.json"
    review_path = out_dir / f"{stem}.review.json"

    if final_path.exists():
        print(f"skip (already done): {pdf_path.name}")
        return

    date, issue = parse_filename(pdf_path.name)
    total_pages = len(PdfReader(str(pdf_path)).pages)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(f"{pdf_path.name}: reusing existing manifest ({len(manifest)} articles)")
    else:
        try:
            manifest = identify_articles(pdf_path, client, model, publication)
        except Exception as e:
            print(f"FAILED (identify step): {pdf_path.name}: {e}")
            return
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"{pdf_path.name}: identified {len(manifest)} articles")

    results, errors, review = [], [], []
    for article in manifest:
        try:
            text, needs_review = extract_article_text(pdf_path, article, client, model, publication, total_pages)
            results.append({
                "filename": pdf_path.name,
                "publication": publication,
                "date": date,
                "issue": issue,
                "article_title": article["article_title"],
                "text": " ".join(text.split()),
            })
            if needs_review:
                review.append({"article_title": article["article_title"], "pages": article["pages"],
                                "reason": "text does not end in terminal punctuation after retry -- possibly still cut off"})
        except Exception as e:
            errors.append({"article_title": article["article_title"], "pages": article["pages"], "error": str(e)})
            print(f"  FAILED: {article['article_title']!r}: {e}")

    final_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    if errors:
        errors_path.write_text(json.dumps(errors, indent=2, ensure_ascii=False))
    if review:
        review_path.write_text(json.dumps(review, indent=2, ensure_ascii=False))
    status = f"{len(results)} articles extracted"
    if errors:
        status += f", {len(errors)} failed (see {errors_path.name})"
    if review:
        status += f", {len(review)} flagged for review (see {review_path.name})"
    print(f"{pdf_path.name}: {status}")

def process_folder(folder, out_dir, model=MODEL, publication=PUBLICATION):
    import anthropic
    client = anthropic.Anthropic()
    for pdf_path in sorted(Path(folder).glob("*.pdf")):
        process_pdf(pdf_path, out_dir, client, model, publication)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python pipeline.py <pdf_folder> <output_folder>")
        sys.exit(1)
    process_folder(sys.argv[1], sys.argv[2])