import os
import sys
import time
sys.path.insert(0, os.path.abspath('.'))

from scripts.collect_data import search_arxiv_papers

os.makedirs('data/raw', exist_ok=True)

print("Searching arXiv (this takes ~30 seconds due to rate limiting)...")
print("Please wait...")

papers = search_arxiv_papers('retrieval augmented generation', max_results=5)

print(f'\nFound {len(papers)} papers')
for p in papers:
    print(f'  - {p["title"][:60]}')