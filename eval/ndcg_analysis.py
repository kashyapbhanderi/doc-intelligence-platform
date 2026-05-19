import os
import sys
import json
sys.path.insert(0, os.path.abspath('.'))


def analyze_ndcg_results(
    baseline_path: str = "eval/ndcg_results.json",
    finetuned_path: str = "eval/ndcg_finetuned.json",
    output_path: str = "eval/ndcg_analysis.json"
):
    """
    Deep analysis of NDCG results comparing
    baseline vs fine-tuned model per query.

    Shows:
    - Which queries improved most
    - Which queries got worse
    - Pattern analysis of failures
    - Difficulty breakdown
    """
    # Load both result files
    if not os.path.exists(baseline_path):
        print(f"Baseline not found: {baseline_path}")
        print("Run: python eval/ndcg_eval.py")
        return

    if not os.path.exists(finetuned_path):
        print(f"Fine-tuned not found: {finetuned_path}")
        print("Run: python eval/compare_models.py")
        return

    with open(baseline_path, encoding='utf-8') as f:
        baseline = json.load(f)
    with open(finetuned_path, encoding='utf-8') as f:
        finetuned = json.load(f)

    base_details = baseline.get("details", [])
    ft_details = finetuned.get("details", [])

    print("=" * 60)
    print("NDCG DEEP ANALYSIS")
    print("=" * 60)

    # Match queries between baseline and fine-tuned
    base_by_q = {
        d["question"]: d for d in base_details}
    ft_by_q = {
        d["question"]: d for d in ft_details}

    common_queries = set(base_by_q.keys()).intersection(
        set(ft_by_q.keys()))

    print(f"Common queries analyzed: {len(common_queries)}")

    improvements = []
    for q in common_queries:
        b = base_by_q[q]
        f = ft_by_q[q]

        hybrid_imp = (
            f["hybrid_ndcg"] - b["hybrid_ndcg"]
        )
        vector_imp = (
            f["vector_ndcg"] - b["vector_ndcg"]
        )

        improvements.append({
            "question": q,
            "source": b.get("source", ""),
            "baseline_hybrid": b["hybrid_ndcg"],
            "finetuned_hybrid": f["hybrid_ndcg"],
            "hybrid_improvement": round(hybrid_imp, 4),
            "baseline_vector": b["vector_ndcg"],
            "finetuned_vector": f["vector_ndcg"],
            "vector_improvement": round(vector_imp, 4),
        })

    # Sort by improvement
    improvements.sort(
        key=lambda x: x["hybrid_improvement"],
        reverse=True
    )

    # Most improved queries
    print("\n📈 TOP 5 MOST IMPROVED QUERIES:")
    print("-" * 60)
    for item in improvements[:5]:
        imp = item["hybrid_improvement"]
        icon = "🟢" if imp > 0 else "🔴"
        print(f"{icon} {imp:+.4f} | "
              f"{item['question'][:55]}")
        print(f"         Base: {item['baseline_hybrid']:.4f} → "
              f"Fine-tuned: {item['finetuned_hybrid']:.4f}")

    # Queries that got worse
    worse = [i for i in improvements
             if i["hybrid_improvement"] < 0]
    print(f"\n📉 QUERIES THAT GOT WORSE: {len(worse)}")
    print("-" * 60)
    for item in worse[:5]:
        print(f"🔴 {item['hybrid_improvement']:+.4f} | "
              f"{item['question'][:55]}")
        print(f"         Base: {item['baseline_hybrid']:.4f} → "
              f"Fine-tuned: {item['finetuned_hybrid']:.4f}")

    # Unchanged queries
    unchanged = [i for i in improvements
                 if i["hybrid_improvement"] == 0]
    print(f"\n⬜ UNCHANGED QUERIES: {len(unchanged)}")

    # Overall stats
    improved = [i for i in improvements
                if i["hybrid_improvement"] > 0]
    print(f"\n📊 SUMMARY:")
    print(f"  Total queries:    {len(improvements)}")
    print(f"  Improved:         {len(improved)} "
          f"({len(improved)/len(improvements)*100:.0f}%)")
    print(f"  Worse:            {len(worse)} "
          f"({len(worse)/len(improvements)*100:.0f}%)")
    print(f"  Unchanged:        {len(unchanged)}")

    if improved:
        avg_imp = sum(
            i["hybrid_improvement"] for i in improved
        ) / len(improved)
        print(f"  Avg improvement:  +{avg_imp:.4f} "
              f"(for queries that improved)")

    # Source analysis
    print(f"\n📚 IMPROVEMENT BY SOURCE DOCUMENT:")
    print("-" * 60)
    source_imp = {}
    for item in improvements:
        src = item["source"]
        if src not in source_imp:
            source_imp[src] = []
        source_imp[src].append(
            item["hybrid_improvement"])

    source_avg = {
        src: sum(vals) / len(vals)
        for src, vals in source_imp.items()
    }
    sorted_sources = sorted(
        source_avg.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("Top 5 most improved sources:")
    for src, avg in sorted_sources[:5]:
        icon = "🟢" if avg > 0 else "🔴"
        print(f"  {icon} {avg:+.4f} | {src}")

    # Save analysis
    analysis = {
        "total_queries": len(improvements),
        "improved_count": len(improved),
        "worse_count": len(worse),
        "unchanged_count": len(unchanged),
        "improvement_rate": round(
            len(improved) / len(improvements) * 100, 1
        ),
        "details": improvements
    }

    with open(output_path, "w",
               encoding='utf-8') as f:
        json.dump(analysis, f,
                  indent=2, ensure_ascii=False)

    print(f"\nFull analysis saved: {output_path}")
    return analysis


if __name__ == "__main__":
    analyze_ndcg_results()