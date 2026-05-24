"""
eval/log_ragas.py
Log RAGAS scores to MLflow.
Creates comparison across all evaluation methods.
"""
import os
import sys
import json
sys.path.insert(0, os.path.abspath('.'))

import mlflow


def log_ragas_results(
    ragas_path: str = "eval/ragas_results.json"
):
    """Log RAGAS scores to MLflow."""
    if not os.path.exists(ragas_path):
        print(f"RAGAS results not found: {ragas_path}")
        print("Run: python eval/ragas_eval.py first")
        return

    with open(ragas_path, encoding='utf-8') as f:
        data = json.load(f)

    scores = data["summary"]
    mlflow.set_experiment("ragas-evaluation")

    with mlflow.start_run(
        run_name="ragas-week6"
    ):
        # Parameters
        mlflow.log_param(
            "model",   "all-MiniLM-L6-v2-finetuned")
        mlflow.log_param(
            "stage",   "week-6")
        mlflow.log_param(
            "samples", scores["num_samples"])
        mlflow.log_param(
            "framework", "RAGAS")

        # Metrics
        mlflow.log_metric(
            "faithfulness",
            scores["faithfulness"])
        mlflow.log_metric(
            "answer_relevancy",
            scores["answer_relevancy"])
        mlflow.log_metric(
            "context_recall",
            scores["context_recall"])
        mlflow.log_metric(
            "context_precision",
            scores["context_precision"])

        # Compute overall
        avg = sum([
            scores["faithfulness"],
            scores["answer_relevancy"],
            scores["context_recall"],
            scores["context_precision"],
        ]) / 4
        mlflow.log_metric("overall_ragas", round(avg, 4))

        # Tags
        mlflow.set_tag("week", "6")
        mlflow.set_tag("eval_type", "RAGAS")

        mlflow.log_artifact(ragas_path)

        run_id = mlflow.active_run().info.run_id
        print(f"RAGAS scores logged to MLflow!")
        print(f"Run ID: {run_id}")
        print(f"\nMetrics:")
        for k, v in scores.items():
            if k != "num_samples":
                print(f"  {k:<25} {v:.4f}")
        print(f"\nView: mlflow ui → ragas-evaluation")


def compare_all_evaluations():
    """
    Print a complete quality report across
    all evaluation methods used in this project.
    """
    report = {}

    # NDCG (retrieval quality)
    if os.path.exists("eval/ndcg_finetuned.json"):
        with open("eval/ndcg_finetuned.json",
                  encoding='utf-8') as f:
            ndcg = json.load(f)
        report["ndcg"] = ndcg["summary"]

    # Answer quality (keyword + source)
    if os.path.exists(
            "eval/answer_eval_finetuned.json"):
        with open(
            "eval/answer_eval_finetuned.json",
            encoding='utf-8'
        ) as f:
            answers = json.load(f)
        report["answers"] = answers["summary"]

    # RAGAS
    if os.path.exists("eval/ragas_results.json"):
        with open("eval/ragas_results.json",
                  encoding='utf-8') as f:
            ragas = json.load(f)
        report["ragas"] = ragas["summary"]

    print("\n" + "=" * 65)
    print("  COMPLETE QUALITY REPORT — ALL EVALUATIONS")
    print("=" * 65)

    if "ndcg" in report:
        n = report["ndcg"]
        print(f"\n  📊 RETRIEVAL QUALITY (NDCG@10)")
        print(f"     Hybrid:  {n['hybrid_ndcg']:.4f}")
        print(f"     Vector:  {n['vector_ndcg']:.4f}")
        print(f"     BM25:    {n['bm25_ndcg']:.4f}")

    if "answers" in report:
        a = report["answers"]
        print(f"\n  🎯 ANSWER QUALITY")
        print(f"     Source accuracy:  "
              f"{a['avg_source_accuracy']:.4f}")
        print(f"     Keyword overlap:  "
              f"{a['avg_keyword_overlap']:.4f}")
        print(f"     Avg latency:      "
              f"{a['avg_latency_seconds']:.2f}s")

    if "ragas" in report:
        r = report["ragas"]
        print(f"\n  🏆 RAGAS SCORES")
        print(f"     Faithfulness:      "
              f"{r['faithfulness']:.4f}")
        print(f"     Answer Relevancy:  "
              f"{r['answer_relevancy']:.4f}")
        print(f"     Context Recall:    "
              f"{r['context_recall']:.4f}")
        print(f"     Context Precision: "
              f"{r['context_precision']:.4f}")

    print("\n" + "=" * 65)
    print("  All metrics logged to MLflow.")
    print("  Run: mlflow ui → open http://localhost:5000")
    print("=" * 65)


if __name__ == "__main__":
    log_ragas_results()
    compare_all_evaluations()