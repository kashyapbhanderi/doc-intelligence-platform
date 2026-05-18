import os
import sys
import json
import random
from pathlib import Path

sys.path.insert(0, os.path.abspath('.'))


def load_all_chunks(
    processed_dir: str = "data/processed"
) -> dict:
    """
    Load all chunks grouped by source document.

    Returns dict: {source_filename: [chunks]}
    Used to find positive and negative examples.
    """
    chunks_by_source = {}
    json_files = list(
        Path(processed_dir).glob("*_processed.json"))

    for json_file in json_files:
        try:
            with open(json_file, encoding='utf-8') as f:
                doc = json.load(f)
            source = doc.get("source", "")
            chunks = [
                c for c in doc.get("chunks", [])
                if len(c.get("text", "")) > 100
            ]
            if chunks:
                chunks_by_source[source] = chunks
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")

    total = sum(len(v) for v in chunks_by_source.values())
    print(f"Loaded {len(chunks_by_source)} documents")
    print(f"Total chunks: {total}")
    return chunks_by_source


def create_triplet_from_qa(
    qa: dict,
    chunks_by_source: dict,
    all_sources: list
) -> dict | None:
    """
    Create one training triplet from a Q&A pair.

    Triplet structure:
    - anchor: the question
    - positive: chunk from the CORRECT source document
    - negative: chunk from a DIFFERENT source document

    The model learns: question should be CLOSER to
    positive than to negative in vector space.

    Args:
        qa: Q&A pair dict
        chunks_by_source: All chunks grouped by source
        all_sources: List of all source filenames

    Returns:
        Triplet dict or None if cannot create
    """
    question = qa.get("question", "")
    source = qa.get("source", "")
    answer = qa.get("answer", "")

    if not question or not source:
        return None

    # Get positive chunk — from correct source
    source_chunks = chunks_by_source.get(source, [])
    if not source_chunks:
        return None

    # Find best positive chunk — one containing answer words
    answer_words = set(answer.lower().split())
    best_positive = None
    best_overlap = 0

    for chunk in source_chunks:
        chunk_words = set(chunk["text"].lower().split())
        overlap = len(answer_words.intersection(
            chunk_words))
        if overlap > best_overlap:
            best_overlap = overlap
            best_positive = chunk

    # Fall back to random chunk from source
    if not best_positive:
        best_positive = random.choice(source_chunks)

    # Get negative chunk — from DIFFERENT source
    # Use a source that is topically similar but wrong
    # This is called a "hard negative"
    other_sources = [
        s for s in all_sources
        if s != source
        and s in chunks_by_source
    ]

    if not other_sources:
        return None

    # Pick random different source
    negative_source = random.choice(other_sources)
    negative_chunks = chunks_by_source[negative_source]
    negative_chunk = random.choice(negative_chunks)

    return {
        "anchor": question,
        "positive": best_positive["text"][:400],
        "negative": negative_chunk["text"][:400],
        "positive_source": source,
        "negative_source": negative_source,
        "overlap_score": best_overlap
    }


def generate_triplets(
    qa_path: str = "eval/qa_dataset.json",
    processed_dir: str = "data/processed",
    target: int = 300,
    output_path: str = "data/triplets.json"
):
    """
    Generate training triplets for embedding fine-tuning.

    Strategy:
    1. Use existing Q&A pairs as anchors (questions)
    2. Find correct chunk from same document (positive)
    3. Find chunk from different document (negative)
    4. Repeat with augmented questions to reach 300

    Args:
        qa_path: Path to Q&A evaluation dataset
        processed_dir: Folder with processed docs
        target: Number of triplets to generate
        output_path: Where to save triplets
    """
    # Load Q&A pairs
    with open(qa_path, encoding='utf-8') as f:
        qa_pairs = json.load(f)

    print(f"Q&A pairs loaded: {len(qa_pairs)}")

    # Load all chunks
    chunks_by_source = load_all_chunks(processed_dir)
    all_sources = list(chunks_by_source.keys())

    print(f"Target triplets: {target}")
    print("=" * 50)

    triplets = []

    # Round 1 — direct from Q&A pairs
    print("Round 1: Creating triplets from Q&A pairs...")
    for qa in qa_pairs:
        if len(triplets) >= target:
            break
        triplet = create_triplet_from_qa(
            qa, chunks_by_source, all_sources)
        if triplet:
            triplets.append(triplet)

    print(f"  After Round 1: {len(triplets)} triplets")

    # Round 2 — augment questions with variations
    print("Round 2: Augmenting with question variations...")
    augment_templates = [
        "Explain the concept of {topic}",
        "What do the authors say about {topic}?",
        "How is {topic} described in the paper?",
        "What is the main finding about {topic}?",
        "Summarize the approach to {topic}",
    ]

    for qa in qa_pairs:
        if len(triplets) >= target:
            break

        # Extract topic from question
        words = qa.get("question", "").split()
        content_words = [
            w for w in words
            if len(w) > 4 and w.isalpha()
        ][:3]

        if not content_words:
            continue

        topic = " ".join(content_words)

        for template in augment_templates:
            if len(triplets) >= target:
                break

            augmented_qa = qa.copy()
            augmented_qa["question"] = \
                template.format(topic=topic)

            triplet = create_triplet_from_qa(
                augmented_qa,
                chunks_by_source,
                all_sources
            )
            if triplet:
                triplets.append(triplet)

    print(f"  After Round 2: {len(triplets)} triplets")

    # Round 3 — chunk-to-chunk triplets
    print("Round 3: Adding chunk-to-chunk triplets...")
    for source, chunks in chunks_by_source.items():
        if len(triplets) >= target:
            break

        # Use first sentence of chunk as anchor
        for chunk in chunks[:3]:
            if len(triplets) >= target:
                break

            text = chunk.get("text", "")
            sentences = [
                s.strip() for s in text.split(".")
                if len(s.strip()) > 30
            ]

            if not sentences:
                continue

            anchor = sentences[0][:200]

            # Positive: another chunk from same doc
            same_chunks = [
                c for c in chunks
                if c != chunk
            ]
            if not same_chunks:
                continue
            positive_chunk = random.choice(same_chunks)

            # Negative: chunk from different doc
            other_sources = [
                s for s in all_sources
                if s != source
            ]
            if not other_sources:
                continue
            neg_source = random.choice(other_sources)
            neg_chunks = chunks_by_source[neg_source]
            negative_chunk = random.choice(neg_chunks)

            triplets.append({
                "anchor": anchor,
                "positive": positive_chunk["text"][:400],
                "negative": negative_chunk["text"][:400],
                "positive_source": source,
                "negative_source": neg_source,
                "overlap_score": 0
            })

    print(f"  After Round 3: {len(triplets)} triplets")

    # Shuffle to mix all rounds
    random.shuffle(triplets)
    triplets = triplets[:target]

    # Save
    os.makedirs(os.path.dirname(output_path)
                if os.path.dirname(output_path)
                else ".", exist_ok=True)

    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(triplets, f, indent=2,
                  ensure_ascii=False)

    # Stats
    print("\n" + "=" * 50)
    print("TRIPLET GENERATION COMPLETE")
    print("=" * 50)
    print(f"Total triplets:     {len(triplets)}")
    print(f"Saved to:           {output_path}")

    # Quality check
    sources_covered = set(
        t["positive_source"] for t in triplets)
    print(f"Sources covered:    {len(sources_covered)}")
    print(f"Avg anchor length:  "
          f"{sum(len(t['anchor']) for t in triplets) // len(triplets)} chars")

    # Show sample
    print(f"\nSample triplet:")
    sample = triplets[0]
    print(f"  Anchor:   {sample['anchor'][:80]}...")
    print(f"  Positive: {sample['positive'][:80]}...")
    print(f"  Negative: {sample['negative'][:80]}...")
    print(f"  Pos src:  {sample['positive_source']}")
    print(f"  Neg src:  {sample['negative_source']}")

    return triplets


if __name__ == "__main__":
    generate_triplets(
        target=300,
        output_path="data/triplets.json"
    )