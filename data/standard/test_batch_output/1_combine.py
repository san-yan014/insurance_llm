
"""
python3 combine_articles.py /path/to/standard_output standard_combined.json
"""

import json
import sys
from pathlib import Path

def load_combined(path):
    if not path.exists():
        return []
    return json.loads(path.read_text())

def combine(input_dir, output_path):
    input_dir, output_path = Path(input_dir), Path(output_path)
    # dedup key, not stored in output
    by_key = {(a["filename"], a["article_title"]): a for a in load_combined(output_path)}

    added = 0
    for f in sorted(input_dir.glob("*_articles.json")):
        for a in json.loads(f.read_text()):
            key = (a["filename"], a["article_title"])
            if key not in by_key:
                added += 1
            by_key[key] = a

    combined = list(by_key.values())
    output_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
    print(f"{len(combined)} total articles, {added} new this run -> {output_path}")

if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "standard_output"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "standard_combined.json"
    combine(input_dir, output_path)