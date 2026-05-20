import os
import sys
import json
sys.path.insert(0, os.path.abspath('.'))

import mlflow
from mlflow.tracking import MlflowClient


def log_complete_week3_summary():
    """
    Log complete Week 3 experiment summary to MLflow.

    Creates one unified run with ALL metrics from
    the entire fine-tuning process:
    - Training configuration
    - NDCG before vs after
    - Answer quality before vs after
    - Improvement percentages
    - Training data stats
    """
    # Load all result files
    results = {}

    # Baseline NDCG
    if os.path.exists("eval/ndcg_results.json"):
        with open("eval/ndcg_results.json",
                  encoding='utf-8') as f:
            results["baseline_ndcg"] = json.load(f)

    # Fine-tuned NDCG
    if os.path.exists("eval/ndcg_finetuned.json"):
        with open("eval/ndcg_finetuned.json",
                  encoding='utf-8') as f:
            results["finetuned_ndcg"] = json.load(f)

    # Baseline answers
    if os.path.exists(
            "eval/answer_eval_results.json"):
        with open("eval/answer_eval_results.json",
                  encoding='utf-8') as f:
            results["baseline_answers"] = \
                json.load(f)

    # Fine-tuned answers
    if os.path.exists(
            "eval/answer_eval_finetuned.json"):
        with open(
            "eval/answer_eval_finetuned.json",
            encoding='utf-8'
        ) as f:
            results["finetuned_answers"] = \
                json.load(f)

    # Triplets
    if os.path.exists("data/triplets_clean.json"):
        with open("data/triplets_clean.json",
                  encoding='utf-8') as f:
            triplets = json.load(f)
        results["num_triplets"] = len(triplets)

    # Analysis
    if os.path.exists("eval/ndcg_analysis.json"):
        with open("eval/ndcg_analysis.json",
                  encoding='utf-8') as f:
            results["analysis"] = json.load(f)

    # Log to MLflow
    mlflow.set_experiment("week3-final-summary")

    with mlflow.start_run(
        run_name="complete-finetuning-results"
    ):
        # Training config
        mlflow.log_param(
            "base_model", "all-MiniLM-L6-v2")
        mlflow.log_param(
            "training_epochs", 3)
        mlflow.log_param(
            "batch_size", 16)
        mlflow.log_param(
            "learning_rate", "2e-5")
        mlflow.log_param(
            "loss_function",
            "MultipleNegativesRankingLoss")
        mlflow.log_param(
            "num_triplets",
            results.get("num_triplets", 274))
        mlflow.log_param(
            "total_documents", 52)
        mlflow.log_param(
            "total_chunks", 11089)

        # NDCG metrics
        if "baseline_ndcg" in results:
            bs = results["baseline_ndcg"]["summary"]
            mlflow.log_metric(
                "baseline_hybrid_ndcg",
                bs["hybrid_ndcg"])
            mlflow.log_metric(
                "baseline_vector_ndcg",
                bs["vector_ndcg"])
            mlflow.log_metric(
                "baseline_bm25_ndcg",
                bs["bm25_ndcg"])

        if "finetuned_ndcg" in results:
            fs = results["finetuned_ndcg"]["summary"]
            mlflow.log_metric(
                "finetuned_hybrid_ndcg",
                fs["hybrid_ndcg"])
            mlflow.log_metric(
                "finetuned_vector_ndcg",
                fs["vector_ndcg"])
            mlflow.log_metric(
                "finetuned_bm25_ndcg",
                fs["bm25_ndcg"])

            # Improvements
            if "baseline_ndcg" in results:
                bs = results[
                    "baseline_ndcg"]["summary"]
                hybrid_imp = (
                    (fs["hybrid_ndcg"] -
                     bs["hybrid_ndcg"]) /
                    bs["hybrid_ndcg"] * 100
                )
                vector_imp = (
                    (fs["vector_ndcg"] -
                     bs["vector_ndcg"]) /
                    bs["vector_ndcg"] * 100
                )
                mlflow.log_metric(
                    "hybrid_ndcg_improvement_pct",
                    round(hybrid_imp, 2))
                mlflow.log_metric(
                    "vector_ndcg_improvement_pct",
                    round(vector_imp, 2))

        # Answer quality metrics
        if "baseline_answers" in results:
            ba = results[
                "baseline_answers"]["summary"]
            mlflow.log_metric(
                "baseline_source_accuracy",
                ba["avg_source_accuracy"])
            mlflow.log_metric(
                "baseline_keyword_overlap",
                ba["avg_keyword_overlap"])
            mlflow.log_metric(
                "baseline_avg_latency",
                ba["avg_latency_seconds"])

        if "finetuned_answers" in results:
            fa = results[
                "finetuned_answers"]["summary"]
            mlflow.log_metric(
                "finetuned_source_accuracy",
                fa["avg_source_accuracy"])
            mlflow.log_metric(
                "finetuned_keyword_overlap",
                fa["avg_keyword_overlap"])
            mlflow.log_metric(
                "finetuned_avg_latency",
                fa["avg_latency_seconds"])

        # Analysis metrics
        if "analysis" in results:
            a = results["analysis"]
            mlflow.log_metric(
                "queries_improved_pct",
                a["improvement_rate"])
            mlflow.log_metric(
                "queries_improved_count",
                a["improved_count"])
            mlflow.log_metric(
                "queries_worse_count",
                a["worse_count"])

        # Tags
        mlflow.set_tag("week", "3")
        mlflow.set_tag("stage", "complete")
        mlflow.set_tag(
            "model",
            "all-MiniLM-L6-v2-finetuned")

        # Log artifacts
        artifacts = [
            "eval/ndcg_results.json",
            "eval/ndcg_finetuned.json",
            "eval/ndcg_analysis.json",
        ]
        for artifact in artifacts:
            if os.path.exists(artifact):
                mlflow.log_artifact(artifact)

        run_id = mlflow.active_run().info.run_id
        print(f"Week 3 summary logged!")
        print(f"Run ID: {run_id}")

    return results


def print_week3_report(results: dict):
    """Print a clean Week 3 summary report."""
    print("\n" + "=" * 60)
    print("WEEK 3 COMPLETE — FINE-TUNING REPORT")
    print("=" * 60)

    print(f"\n📊 TRAINING CONFIGURATION")
    print(f"  Base model:     all-MiniLM-L6-v2")
    print(f"  Training data:  {results.get('num_triplets', 274)} triplets")
    print(f"  Epochs:         3")
    print(f"  Loss function:  MultipleNegativesRankingLoss")

    if ("baseline_ndcg" in results and
            "finetuned_ndcg" in results):
        bs = results["baseline_ndcg"]["summary"]
        fs = results["finetuned_ndcg"]["summary"]

        hybrid_imp = (
            (fs["hybrid_ndcg"] - bs["hybrid_ndcg"])
            / bs["hybrid_ndcg"] * 100
        )

        print(f"\n📈 NDCG@10 IMPROVEMENT")
        print(f"  Baseline hybrid:    "
              f"{bs['hybrid_ndcg']:.4f}")
        print(f"  Fine-tuned hybrid:  "
              f"{fs['hybrid_ndcg']:.4f}")
        print(f"  Improvement:        "
              f"+{hybrid_imp:.1f}%")

    if ("baseline_answers" in results and
            "finetuned_answers" in results):
        ba = results["baseline_answers"]["summary"]
        fa = results["finetuned_answers"]["summary"]

        src_imp = (
            (fa["avg_source_accuracy"] -
             ba["avg_source_accuracy"]) /
            max(ba["avg_source_accuracy"], 0.001) * 100
        )

        print(f"\n🎯 ANSWER QUALITY IMPROVEMENT")
        print(f"  Baseline source accuracy:   "
              f"{ba['avg_source_accuracy']:.4f}")
        print(f"  Fine-tuned source accuracy: "
              f"{fa['avg_source_accuracy']:.4f}")
        print(f"  Improvement:                "
              f"+{src_imp:.1f}%")

    if "analysis" in results:
        a = results["analysis"]
        print(f"\n🔍 QUERY ANALYSIS")
        print(f"  Total queries:   "
              f"{a['total_queries']}")
        print(f"  Improved:        "
              f"{a['improved_count']} "
              f"({a['improvement_rate']:.0f}%)")
        print(f"  Worse:           "
              f"{a['worse_count']}")

    print(f"\n⭐ RESUME BULLET POINT:")
    print("-" * 60)
    if ("baseline_ndcg" in results and
            "finetuned_ndcg" in results):
        bs = results["baseline_ndcg"]["summary"]
        fs = results["finetuned_ndcg"]["summary"]
        imp = (
            (fs["hybrid_ndcg"] - bs["hybrid_ndcg"])
            / bs["hybrid_ndcg"] * 100
        )
        print(
            f"Fine-tuned sentence-transformer embedding\n"
            f"model using MultipleNegativesRankingLoss on\n"
            f"{results.get('num_triplets', 274)} domain "
            f"triplets, improving retrieval\n"
            f"NDCG@10 from {bs['hybrid_ndcg']:.4f} to "
            f"{fs['hybrid_ndcg']:.4f} (+{imp:.1f}%)"
        )
    print("-" * 60)

    print(f"\n✅ Week 3 Complete!")
    print(f"   Next: Week 4 — Multi-agent RAG System")


if __name__ == "__main__":
    results = log_complete_week3_summary()
    print_week3_report(results)