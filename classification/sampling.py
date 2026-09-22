"""
Usage:

    python sampling.py --rough_notes_input full_corpus_with_urls.json --standard_input standard_filtered.json --rough_notes 40 --standard 10 --output batches/batch_1.json --classified classified_articles_list.json

"""
import json
import random
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--rough_notes_input', required=True, help='Rough notes corpus JSON')
parser.add_argument('--standard_input', required=True, help='Standard corpus JSON')
parser.add_argument('--rough_notes', type=int, required=True, help='Number of rough notes articles to sample')
parser.add_argument('--standard', type=int, required=True, help='Number of standard articles to sample')
parser.add_argument('--output', required=True, help='Output batch JSON file')
parser.add_argument('--classified', default='classified_articles_list.json', help='Path to classified articles list JSON')
args = parser.parse_args()

with open(args.rough_notes_input, 'r', encoding='utf-8') as f:
    rough_notes = json.load(f)

with open(args.standard_input, 'r', encoding='utf-8') as f:
    standard = json.load(f)

already_classified = set()
if os.path.exists(args.classified):
    with open(args.classified, 'r', encoding='utf-8') as f:
        classified = json.load(f)
    for article in classified:
        uid = article.get('GOID') or article.get('filename') or article.get('title')
        if uid:
            already_classified.add(uid)
    print(f"Loaded {len(already_classified)} already-classified articles to exclude")
else:
    print(f"No classified articles list found at {args.classified}, proceeding without exclusions")

def get_uid(article):
    return article.get('GOID') or article.get('filename') or article.get('title')

rough_notes = [a for a in rough_notes if get_uid(a) not in already_classified]
standard = [a for a in standard if get_uid(a) not in already_classified]

print(f"Rough notes available after exclusion: {len(rough_notes)}")
print(f"Standard available after exclusion: {len(standard)}")

all_keys = set()
for article in rough_notes:
    all_keys.update(article.keys())

normalized_standard = []
for article in standard:
    normalized = {key: article.get(key, '') for key in all_keys}
    normalized_standard.append(normalized)

standard = normalized_standard

if args.rough_notes > len(rough_notes):
    print(f"WARNING: requested {args.rough_notes} rough notes articles but only {len(rough_notes)} available. Using all.")
    args.rough_notes = len(rough_notes)

if args.standard > len(standard):
    print(f"WARNING: requested {args.standard} standard articles but only {len(standard)} available. Using all.")
    args.standard = len(standard)

random.shuffle(rough_notes)
random.shuffle(standard)

sampled = rough_notes[:args.rough_notes] + standard[:args.standard]
random.shuffle(sampled)

os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None

with open(args.output, 'w', encoding='utf-8') as f:
    json.dump(sampled, f, ensure_ascii=False, indent=2)

print(f"Sampled {args.rough_notes} rough notes + {args.standard} standard = {len(sampled)} total articles")
print(f"Saved to {args.output}")