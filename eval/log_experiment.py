import os
import sys
import json
sys.path.insert(0, os.path.abspath('.'))

import mlflow


def log_baseline_experiment(
    results_path: str = "eval/ndcg_results.json"
):
    """
    Log baseline retrieval results to MLflow.

    MLflow tracks:
    - Parameters: what model/config was used
    - Metrics: the NDCG scores
    - Tags: useful labels for filtering

    This creates your experiment history that
    grows throughout the project.
    """
    if not os.path.exists(results_path):
        print(f"Results not found: {results_path}")
        print("Run: python eval/ndcg_eval.py first")
        return

    with open(results_path, encoding='utf-8') as f:
        results = json.load(f)

    summary = results["summary"]

    # Set experiment name
    mlflow.set_experiment("retrieval-quality")

    with mlflow.start_run(run_name="baseline-all-MiniLM"):

        # Log parameters (what config was used)
        mlflow.log_param("model",
                         "all-MiniLM-L6-v2")
        mlflow.log_param("vector_dim", 384)
        mlflow.log_param("chunk_size", 512)
        mlflow.log_param("chunk_overlap", 50)
        mlflow.log_param("num_documents", 52)
        mlflow.log_param("total_chunks", 11089)
        mlflow.log_param("k", summary["k"])
        mlflow.log_param("stage", "baseline")

        # Log metrics (the scores)
        mlflow.log_metric("bm25_ndcg",
                          summary["bm25_ndcg"])
        mlflow.log_metric("vector_ndcg",
                          summary["vector_ndcg"])
        mlflow.log_metric("hybrid_ndcg",
                          summary["hybrid_ndcg"])

        # Log tags
        mlflow.set_tag("week", "2")
        mlflow.set_tag("model_type", "baseline")
        mlflow.set_tag("note",
                       "Before fine-tuning — Week 3 target")

        # Log the results file as artifact
        mlflow.log_artifact(results_path)

        run_id = mlflow.active_run().info.run_id
        print(f"Logged to MLflow!")
        print(f"Run ID: {run_id}")
        print(f"Experiment: retrieval-quality")
        print(f"\nMetrics logged:")
        print(f"  BM25 NDCG@10:   "
              f"{summary['bm25_ndcg']:.4f}")
        print(f"  Vector NDCG@10: "
              f"{summary['vector_ndcg']:.4f}")
        print(f"  Hybrid NDCG@10: "
              f"{summary['hybrid_ndcg']:.4f}")
        print(f"\nView dashboard:")
        print(f"  Run: mlflow ui")
        print(f"  Open: http://localhost:5000")


if __name__ == "__main__":
    log_baseline_experiment()