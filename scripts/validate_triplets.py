import os
import sys
import json
sys.path.insert(0, os.path.abspath('.'))


def validate_triplets(
    triplets_path: str = "data/triplets.json",
    output_path: str = "data/triplets_clean.json"
):
    """
    Validate and clean triplets before fine-tuning.

    Removes triplets that are:
    - Too short (not enough content)
    - Positive and negative from same source
    - Anchor same as positive (trivial)
    - Any field missing

    Args:
        triplets_path: Raw triplets file
        output_path: Cleaned triplets output

    Returns:
        List of clean triplets
    """
    with open(triplets_path, encoding='utf-8') as f:
        triplets = json.load(f)

    print(f"Input triplets:  {len(triplets)}")
    print("Validating...")

    clean = []
    removed = {
        "missing_fields": 0,
        "too_short": 0,
        "same_source": 0,
        "duplicate_anchor": 0,
        "trivial": 0
    }

    seen_anchors = set()

    for t in triplets:
        anchor = t.get("anchor", "")
        positive = t.get("positive", "")
        negative = t.get("negative", "")
        pos_src = t.get("positive_source", "")
        neg_src = t.get("negative_source", "")

        # Check all fields present
        if not all([anchor, positive, negative]):
            removed["missing_fields"] += 1
            continue

        # Check minimum length
        if (len(anchor) < 10 or
                len(positive) < 50 or
                len(negative) < 50):
            removed["too_short"] += 1
            continue

        # Check sources are different
        if pos_src == neg_src:
            removed["same_source"] += 1
            continue

        # Check anchor not same as positive
        if anchor[:50] == positive[:50]:
            removed["trivial"] += 1
            continue

        # Check no duplicate anchors
        anchor_key = anchor[:50].lower()
        if anchor_key in seen_anchors:
            removed["duplicate_anchor"] += 1
            continue
        seen_anchors.add(anchor_key)

        clean.append(t)

    # Save clean triplets
    with open(output_path, "w",
               encoding='utf-8') as f:
        json.dump(clean, f, indent=2,
                  ensure_ascii=False)

    # Print report
    print("\n" + "=" * 50)
    print("VALIDATION REPORT")
    print("=" * 50)
    print(f"Input:           {len(triplets)}")
    print(f"Clean:           {len(clean)}")
    print(f"Removed:         {len(triplets)-len(clean)}")
    print(f"\nRemoval reasons:")
    for reason, count in removed.items():
        if count > 0:
            print(f"  {reason}: {count}")

    print(f"\nQuality checks passed:")
    print(f"  All fields present:     ✅")
    print(f"  Minimum length:         ✅")
    print(f"  Different sources:      ✅")
    print(f"  Unique anchors:         ✅")
    print(f"  No trivial pairs:       ✅")

    print(f"\nClean triplets: {len(clean)}")
    print(f"Saved to: {output_path}")

    if len(clean) >= 200:
        print("\n✅ Enough triplets for fine-tuning!")
        print("   Week 3 fine-tuning is ready to start.")
    else:
        print(f"\n⚠️  Only {len(clean)} clean triplets.")
        print("   Minimum 200 recommended.")

    return clean


def show_triplet_stats(
    triplets_path: str = "data/triplets_clean.json"
):
    """Show statistics about the clean triplets."""
    with open(triplets_path, encoding='utf-8') as f:
        triplets = json.load(f)

    # Source coverage
    sources = set(t["positive_source"]
                  for t in triplets)
    print(f"\nSource coverage: {len(sources)} papers")

    # Length stats
    anchor_lens = [len(t["anchor"]) for t in triplets]
    pos_lens = [len(t["positive"]) for t in triplets]
    neg_lens = [len(t["negative"]) for t in triplets]

    print(f"\nLength statistics:")
    print(f"  Anchor:   avg={sum(anchor_lens)//len(anchor_lens)} chars")
    print(f"  Positive: avg={sum(pos_lens)//len(pos_lens)} chars")
    print(f"  Negative: avg={sum(neg_lens)//len(neg_lens)} chars")

    # Top sources
    from collections import Counter
    source_counts = Counter(
        t["positive_source"] for t in triplets)
    print(f"\nTop 5 sources by triplet count:")
    for src, count in source_counts.most_common(5):
        print(f"  {src}: {count}")


if __name__ == "__main__":
    # Validate
    clean = validate_triplets()

    # Show stats
    if os.path.exists("data/triplets_clean.json"):
        show_triplet_stats()