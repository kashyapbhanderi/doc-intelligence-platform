import os
import sys
import json
sys.path.insert(0, os.path.abspath('.'))

from eval.ndcg_eval import evaluate_retrieval
import mlflow


def compare_base_vs_finetuned(
    qa_path: str = "eval/qa_dataset.json",
    baseline_path: str = "eval/ndcg_results.json",
    finetuned_path: str = "eval/ndcg_finetuned.json"
):
    """
    Compare NDCG scores: baseline vs fine-tuned model.

    This is the KEY measurement of Week 3.
    The improvement % goes directly on your resume.
    """
    print("=" * 60)
    print("MODEL COMPARISON: BASE vs FINE-TUNED")
    print("=" * 60)

    # Load baseline results
    if os.path.exists(baseline_path):
        with open(baseline_path,
                  encoding='utf-8') as f:
            baseline = json.load(f)
        base_scores = baseline["summary"]
        print(f"\nBaseline scores already loaded")
    else:
        print("Baseline not found. Run ndcg_eval first.")
        return

    # Run evaluation with fine-tuned model
    print("\nRunning evaluation with fine-tuned model...")
    print("(Uses current Weaviate — must be re-embedded)\n")

    ft_results = evaluate_retrieval(
        qa_path=qa_path,
        output_path=finetuned_path
    )

    if not ft_results:
        print("Evaluation failed!")
        return

    ft_scores = ft_results["summary"]

    # Calculate improvements
    bm25_imp = (
        (ft_scores["bm25_ndcg"] -
         base_scores["bm25_ndcg"]) /
        base_scores["bm25_ndcg"] * 100
    )
    vector_imp = (
        (ft_scores["vector_ndcg"] -
         base_scores["vector_ndcg"]) /
        base_scores["vector_ndcg"] * 100
    )
    hybrid_imp = (
        (ft_scores["hybrid_ndcg"] -
         base_scores["hybrid_ndcg"]) /
        base_scores["hybrid_ndcg"] * 100
    )

    # Print comparison table
    print("\n" + "=" * 60)
    print("NDCG@10 COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Method':<12} {'Baseline':>10} "
          f"{'Fine-tuned':>12} {'Improvement':>13}")
    print("-" * 50)
    print(f"{'BM25':<12} "
          f"{base_scores['bm25_ndcg']:>10.4f} "
          f"{ft_scores['bm25_ndcg']:>12.4f} "
          f"{bm25_imp:>+12.1f}%")
    print(f"{'Vector':<12} "
          f"{base_scores['vector_ndcg']:>10.4f} "
          f"{ft_scores['vector_ndcg']:>12.4f} "
          f"{vector_imp:>+12.1f}%")
    print(f"{'Hybrid':<12} "
          f"{base_scores['hybrid_ndcg']:>10.4f} "
          f"{ft_scores['hybrid_ndcg']:>12.4f} "
          f"{hybrid_imp:>+12.1f}%")
    print("=" * 60)

    # Resume bullet point
    print("\n⭐ YOUR RESUME BULLET POINT:")
    print("-" * 60)
    print(
        f"Fine-tuned all-MiniLM-L6-v2 embedding model "
        f"on {274} domain-specific triplets using "
        f"MultipleNegativesRankingLoss, improving "
        f"retrieval NDCG@10 from "
        f"{base_scores['hybrid_ndcg']:.4f} to "
        f"{ft_scores['hybrid_ndcg']:.4f} "
        f"(+{hybrid_imp:.1f}%)"
    )
    print("-" * 60)

    # Log to MLflow
    mlflow.set_experiment("model-comparison")
    with mlflow.start_run(
        run_name="base-vs-finetuned"
    ):
        # Baseline metrics
        mlflow.log_metric(
            "baseline_bm25_ndcg",
            base_scores["bm25_ndcg"]
        )
        mlflow.log_metric(
            "baseline_vector_ndcg",
            base_scores["vector_ndcg"]
        )
        mlflow.log_metric(
            "baseline_hybrid_ndcg",
            base_scores["hybrid_ndcg"]
        )

        # Fine-tuned metrics
        mlflow.log_metric(
            "finetuned_bm25_ndcg",
            ft_scores["bm25_ndcg"]
        )
        mlflow.log_metric(
            "finetuned_vector_ndcg",
            ft_scores["vector_ndcg"]
        )
        mlflow.log_metric(
            "finetuned_hybrid_ndcg",
            ft_scores["hybrid_ndcg"]
        )

        # Improvement metrics
        mlflow.log_metric(
            "hybrid_ndcg_improvement_pct",
            round(hybrid_imp, 2)
        )

        mlflow.set_tag("week", "3")
        mlflow.set_tag("stage", "comparison")

    print("\nResults logged to MLflow!")
    print("Run: mlflow ui")
    print("Open: http://localhost:5000")

    return {
        "baseline": base_scores,
        "finetuned": ft_scores,
        "improvements": {
            "bm25": round(bm25_imp, 2),
            "vector": round(vector_imp, 2),
            "hybrid": round(hybrid_imp, 2)
        }
    }


if __name__ == "__main__":
    compare_base_vs_finetuned()