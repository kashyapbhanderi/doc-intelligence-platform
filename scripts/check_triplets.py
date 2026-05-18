import json
import os

path = "data/triplets_clean.json"

if not os.path.exists(path):
    print("Clean triplets not found!")
    print("Run: python scripts/validate_triplets.py")
else:
    with open(path, encoding='utf-8') as f:
        triplets = json.load(f)

    print(f"Clean triplets ready: {len(triplets)}")
    print(f"Minimum needed:       200")

    if len(triplets) >= 200:
        print("Status: ✅ Ready for fine-tuning!")
    else:
        print("Status: ❌ Need more triplets")

    # Show sample
    print(f"\nSample triplet:")
    t = triplets[0]
    print(f"  Anchor:   {t['anchor'][:80]}")
    print(f"  Positive: {t['positive'][:80]}...")
    print(f"  Negative: {t['negative'][:80]}...")
    print(f"  Pos src:  {t['positive_source']}")
    print(f"  Neg src:  {t['negative_source']}")