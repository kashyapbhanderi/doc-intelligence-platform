import os
import sys
import json
sys.path.insert(0, os.path.abspath('.'))

import mlflow


def log_retrieval_results(
    ndcg_path: str = "eval/ndcg_results.json",
    run_name: str = "baseline-all-MiniLM"
):
    """Log NDCG retrieval results to MLflow."""
    if not os.path.exists(ndcg_path):
        print(f"NDCG results not found: {ndcg_path}")
        print("Run: python eval/ndcg_eval.py first")
        return None

    with open(ndcg_path, encoding='utf-8') as f:
        results = json.load(f)

    summary = results["summary"]
    mlflow.set_experiment("retrieval-quality")

    with mlflow.start_run(run_name=run_name):
        # Parameters
        mlflow.log_param("model", summary.get(
            "model", "all-MiniLM-L6-v2"))
        mlflow.log_param("vector_dim", 384)
        mlflow.log_param("chunk_size", 512)
        mlflow.log_param("chunk_overlap", 50)
        mlflow.log_param("num_documents", 52)
        mlflow.log_param("total_chunks", 11089)
        mlflow.log_param("k", summary["k"])
        mlflow.log_param("stage", "baseline")

        # Metrics
        mlflow.log_metric("bm25_ndcg",
                          summary["bm25_ndcg"])
        mlflow.log_metric("vector_ndcg",
                          summary["vector_ndcg"])
        mlflow.log_metric("hybrid_ndcg",
                          summary["hybrid_ndcg"])

        # Tags
        mlflow.set_tag("week", "2")
        mlflow.set_tag("model_type", "baseline")
        mlflow.set_tag("note",
                       "Baseline before fine-tuning")

        # Artifacts
        mlflow.log_artifact(ndcg_path)

        run_id = mlflow.active_run().info.run_id
        print(f"Logged retrieval results!")
        print(f"Run ID: {run_id}")
        return run_id


def log_answer_results(
    answer_path: str = "eval/answer_eval_results.json",
    run_name: str = "baseline-answer-quality"
):
    """Log answer quality results to MLflow."""
    if not os.path.exists(answer_path):
        print(f"Answer results not found: {answer_path}")
        print("Run: python eval/answer_eval.py first")
        return None

    with open(answer_path, encoding='utf-8') as f:
        results = json.load(f)

    summary = results["summary"]
    mlflow.set_experiment("answer-quality")

    with mlflow.start_run(run_name=run_name):
        # Parameters
        mlflow.log_param("model",
                         summary.get("model",
                                     "all-MiniLM-L6-v2"))
        mlflow.log_param("stage", "baseline")
        mlflow.log_param("questions_evaluated",
                         summary["total_evaluated"])

        # Metrics
        mlflow.log_metric(
            "keyword_overlap",
            summary["avg_keyword_overlap"]
        )
        mlflow.log_metric(
            "source_accuracy",
            summary["avg_source_accuracy"]
        )
        mlflow.log_metric(
            "avg_latency",
            summary["avg_latency_seconds"]
        )
        mlflow.log_metric(
            "p95_latency",
            summary["p95_latency_seconds"]
        )

        # Tags
        mlflow.set_tag("week", "2")
        mlflow.set_tag("model_type", "baseline")

        # Artifacts
        mlflow.log_artifact(answer_path)

        run_id = mlflow.active_run().info.run_id
        print(f"Logged answer quality results!")
        print(f"Run ID: {run_id}")
        return run_id


def print_all_experiments():
    """Print summary of all logged experiments."""
    client = mlflow.tracking.MlflowClient()

    print("\n" + "=" * 60)
    print("ALL MLFLOW EXPERIMENTS")
    print("=" * 60)

    experiments = client.search_experiments()
    for exp in experiments:
        if exp.name == "Default":
            continue

        print(f"\nExperiment: {exp.name}")
        print("-" * 40)

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"]
        )

        for run in runs[:5]:
            print(f"  Run: {run.info.run_name}")
            for key, value in run.data.metrics.items():
                print(f"    {key}: {value:.4f}")


if __name__ == "__main__":
    print("Logging all experiment results to MLflow...")
    print("=" * 50)

    # Log retrieval results
    print("\n1. Logging retrieval (NDCG) results...")
    log_retrieval_results()

    # Log answer quality results
    print("\n2. Logging answer quality results...")
    log_answer_results()

    # Print summary
    print_all_experiments()

    print("\n" + "=" * 50)
    print("All results logged!")
    print("View dashboard: mlflow ui")
    print("Then open: http://localhost:5000")