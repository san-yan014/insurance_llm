import json
import random
import sys
from pathlib import Path

def sample_articles(input_path, output_path, n=150, seed=42):
    input_path, output_path = Path(input_path), Path(output_path)
    articles = json.loads(input_path.read_text())

    if len(articles) < n:
        raise ValueError(f"only {len(articles)} articles available, need {n}")

    random.seed(seed)
    sample = random.sample(articles, n)

    output_path.write_text(json.dumps(sample, indent=2, ensure_ascii=False))
    print(f"sampled {n} of {len(articles)} articles -> {output_path}")

if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "combined.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "sample_150.json"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    sample_articles(input_path, output_path, n)