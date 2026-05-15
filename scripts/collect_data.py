import os
import time
import json
import requests
import arxiv
from tqdm import tqdm
from pathlib import Path


# ── Configuration ──────────────────────────────────────────────────
DOWNLOAD_DIR = "data/raw"
METADATA_FILE = "data/dataset_metadata.json"
TARGET_COUNT = 100
DELAY_BETWEEN_DOWNLOADS = 1.5  # seconds — be polite to servers

# Search topics — AI/ML papers (relevant to your project domain)
SEARCH_TOPICS = [
    "large language models",
    "retrieval augmented generation",
    "document understanding",
    "natural language processing",
    "question answering"
]


# ───────────────────────────────────────────────────────────────────


def search_arxiv_papers(topic: str, max_results: int = 20) -> list:
    """
    Search arXiv for papers on a topic.
    Uses delays to avoid rate limiting (HTTP 429).
    """
    import time

    # arxiv Client with built-in delay between requests
    client = arxiv.Client(
        page_size=10,          # fetch 10 at a time (not 100)
        delay_seconds=5,       # wait 5 seconds between requests
        num_retries=3          # retry 3 times if fails
    )

    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    try:
        for result in client.results(search):
            papers.append({
                "id": result.entry_id.split("/")[-1],
                "title": result.title,
                "authors": [a.name for a in result.authors[:3]],
                "abstract": result.summary[:300],
                "topic": topic,
                "pdf_url": result.pdf_url,
                "published": str(result.published.date())
            })
            time.sleep(1)  # extra 1 second between each result

    except Exception as e:
        print(f"  Search error for '{topic}': {e}")
        print("  Waiting 30 seconds before continuing...")
        time.sleep(30)  # wait longer if error occurs

    return papers


def download_pdf(url: str, save_path: str) -> bool:
    """
    Download a single PDF from URL.
    Returns True if successful, False if failed.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Research Project)"}
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)

            # Verify file is actually a PDF and not too small
            size = os.path.getsize(save_path)
            if size < 5000:  # less than 5KB is probably an error page
                os.remove(save_path)
                return False
            return True
        else:
            return False

    except requests.Timeout:
        print(f"  Timeout downloading {url}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def collect_dataset(target: int = TARGET_COUNT):
    """
    Main collection function.
    Downloads PDFs across all topics until target count reached.
    """
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Load existing metadata if any
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE) as f:
            metadata = json.load(f)
        print(f"Resuming — already have {len(metadata)} papers")
    else:
        metadata = {}

    print(f"\nTarget: {target} PDFs")
    print(f"Topics: {len(SEARCH_TOPICS)}")
    print(f"Save directory: {DOWNLOAD_DIR}")
    print("=" * 50)

    total_downloaded = len(metadata)

    for topic in SEARCH_TOPICS:
        if total_downloaded >= target:
            print(f"\nTarget of {target} reached!")
            break

        print(f"\nSearching: '{topic}'")
        papers = search_arxiv_papers(topic, max_results=60)
        print(f"Found {len(papers)} papers")

        # Filter out already downloaded
        new_papers = [
            p for p in papers
            if p["id"] not in metadata
        ]
        print(f"New papers to download: {len(new_papers)}")

        for paper in tqdm(new_papers, desc=f"Downloading"):
            if total_downloaded >= target:
                break

            paper_id = paper["id"]
            save_path = f"{DOWNLOAD_DIR}/{paper_id}.pdf"

            # Skip if already exists on disk
            if os.path.exists(save_path):
                metadata[paper_id] = paper
                total_downloaded += 1
                continue

            # Download
            success = download_pdf(paper["pdf_url"], save_path)

            if success:
                file_size = os.path.getsize(save_path)
                paper["file_path"] = save_path
                paper["file_size_kb"] = round(file_size / 1024, 1)
                paper["status"] = "downloaded"
                metadata[paper_id] = paper
                total_downloaded += 1
            else:
                paper["status"] = "failed"
                metadata[paper_id] = paper

            # Save metadata after every download
            with open(METADATA_FILE, "w") as f:
                json.dump(metadata, f, indent=2)

            # Be polite — wait between requests
            time.sleep(DELAY_BETWEEN_DOWNLOADS)

        print(f"Progress: {total_downloaded}/{target} PDFs")

    # Final summary
    successful = [
        p for p in metadata.values()
        if p.get("status") == "downloaded"
    ]
    failed = [
        p for p in metadata.values()
        if p.get("status") == "failed"
    ]

    print("\n" + "=" * 50)
    print(f"Collection complete!")
    print(f"Successfully downloaded: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Metadata saved: {METADATA_FILE}")

    return metadata


if __name__ == "__main__":
    collect_dataset(target=TARGET_COUNT)