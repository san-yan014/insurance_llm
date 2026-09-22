"""
For each PDF listed in failed_pdfs.txt, writes a stub {stem}_articles.json with
`count` blank article records (title/text empty, ready for manual fill-in) instead
of running extraction. Sits in the normal output folder alongside real extraction
output, same schema, so combine_articles.py and classification.py both work on it
without any special-casing.

Note: since this writes the same {stem}_articles.json filename the real pipeline
checks for "already done," it will block normal (re-)extraction for these PDFs
until the stub file is deleted -- that's the point, since this is an explicit
opt-out of automation for these volumes.

Usage:
    python make_placeholder_articles.py failed_pdfs.txt output_folder [count]
"""
import json
import re
import sys
from pathlib import Path

PUBLICATION = "The Standard"

def parse_filename(filename):
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})_(.+)\.pdf$", filename)
    if not m:
        return None, None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}", m.group(4)

# each entry gets a distinct title -- if they were all blank/identical, combine_articles.py's
# (filename, article_title) dedup key would collapse all of them into just one entry
def make_placeholder_entries(filename, count):
    date, issue = parse_filename(filename)
    return [
        {
            "filename": filename,
            "publication": PUBLICATION,
            "date": date,
            "issue": issue,
            "article_title": "",
            "text": "",
        }
        for i in range(count)
    ]

def make_placeholders(failed_list_path, out_dir, count=10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = [line.strip() for line in Path(failed_list_path).read_text().splitlines() if line.strip()]
    for name in names:
        stem = Path(name).stem
        out_path = out_dir / f"{stem}_articles.json"
        if out_path.exists():
            print(f"skip (already has output): {name}")
            continue
        entries = make_placeholder_entries(name, count)
        out_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
        print(f"{name}: wrote {count} placeholder entries -> {out_path.name}")

if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("usage: python make_placeholder_articles.py <failed_list.txt> <output_folder> [count]")
        sys.exit(1)
    failed_list, out_dir = sys.argv[1], sys.argv[2]
    count = int(sys.argv[3]) if len(sys.argv) == 4 else 10
    make_placeholders(failed_list, out_dir, count)