"""
Fills in one placeholder entry's title and text in a stub _articles.json --
paste the raw article text (line-wraps and all), it gets normalized the same
way the real pipeline does, and written back out as valid JSON automatically.
You never touch JSON syntax by hand.

Usage:
    python fill_placeholder.py <articles.json> <entry_number> "<article title>"
    then paste the article text, then press Ctrl-D (Ctrl-Z, Enter on Windows) to finish
"""
import json
import sys
from pathlib import Path

def fill_entry(path, entry_number, title, raw_text):
    path = Path(path)
    entries = json.loads(path.read_text())

    idx = entry_number - 1
    if not (0 <= idx < len(entries)):
        raise ValueError(f"entry {entry_number} out of range (file has {len(entries)} entries)")

    entries[idx]["article_title"] = title
    entries[idx]["text"] = " ".join(raw_text.split())  # collapses PDF line-wraps into one clean string

    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    print(f"entry {entry_number} filled in: {title!r} ({len(entries[idx]['text'])} chars)")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('usage: python fill_placeholder.py <articles.json> <entry_number> "<article title>"')
        print("then paste the article text, then press Ctrl-D (Ctrl-Z, Enter on Windows) to finish")
        sys.exit(1)
    path, entry_number, title = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    raw_text = sys.stdin.read()
    fill_entry(path, entry_number, title, raw_text)