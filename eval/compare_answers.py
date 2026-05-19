import os
import sys
import json
sys.path.insert(0, os.path.abspath('.'))

from eval.answer_eval import evaluate_answers
import mlflow


def compare_answer_quality():
    """
    Compare answer quality: baseline vs fine-tuned.
    Re-runs answer evaluation on current Weaviate
    which now has fine-tuned embeddings.
    """
    print("=" * 60)
    print("ANSWER QUALITY COMPARISON")
    print("=" * 60)

    # Load baseline answer results
    baseline_path = "eval/answer_eval_results.json"
    finetuned_path = "eval/answer_eval_finetuned.json"

    if os.path.exists(baseline_path):
        with open(baseline_path,
                  encoding='utf-8') as f:
            baseline = json.load(f)
        base_summary = baseline["summary"]
        print(f"Baseline results loaded")
        print(f"  Keyword overlap: "
              f"{base_summary['avg_keyword_overlap']:.4f}")
        print(f"  Source accuracy: "
              f"{base_summary['avg_source_accuracy']:.4f}")
        print(f"  Avg latency:     "
              f"{base_summary['avg_latency_seconds']:.2f}s")
    else:
        print("Baseline not found. Skipping comparison.")
        base_summary = None

    # Run fresh evaluation
    print(f"\nRunning evaluation with fine-tuned model...")
    ft_summary = evaluate_answers(
        output_path=finetuned_path,
        limit=20
    )

    if not ft_summary:
        print("Evaluation failed!")
        return

    # Print comparison
    if base_summary:
        kw_imp = (
            (ft_summary["avg_keyword_overlap"] -
             base_summary["avg_keyword_overlap"]) /
            max(base_summary["avg_keyword_overlap"],
                0.001) * 100
        )
        src_imp = (
            (ft_summary["avg_source_accuracy"] -
             base_summary["avg_source_accuracy"]) /
            max(base_summary["avg_source_accuracy"],
                0.001) * 100
        )

        print("\n" + "=" * 60)
        print("ANSWER QUALITY COMPARISON")
        print("=" * 60)
        print(f"{'Metric':<25} {'Baseline':>10} "
              f"{'Fine-tuned':>12} {'Change':>8}")
        print("-" * 58)
        print(f"{'Keyword overlap':<25} "
              f"{base_summary['avg_keyword_overlap']:>10.4f} "
              f"{ft_summary['avg_keyword_overlap']:>12.4f} "
              f"{kw_imp:>+7.1f}%")
        print(f"{'Source accuracy':<25} "
              f"{base_summary['avg_source_accuracy']:>10.4f} "
              f"{ft_summary['avg_source_accuracy']:>12.4f} "
              f"{src_imp:>+7.1f}%")
        print(f"{'Avg latency (s)':<25} "
              f"{base_summary['avg_latency_seconds']:>10.2f} "
              f"{ft_summary['avg_latency_seconds']:>12.2f} "
              f"{'':>8}")
        print("=" * 60)

        # Log to MLflow
        mlflow.set_experiment("answer-quality")
        with mlflow.start_run(
            run_name="finetuned-answer-quality"
        ):
            mlflow.log_metric(
                "keyword_overlap",
                ft_summary["avg_keyword_overlap"]
            )
            mlflow.log_metric(
                "source_accuracy",
                ft_summary["avg_source_accuracy"]
            )
            mlflow.log_metric(
                "avg_latency",
                ft_summary["avg_latency_seconds"]
            )
            mlflow.log_metric(
                "keyword_improvement_pct",
                round(kw_imp, 2)
            )
            mlflow.log_metric(
                "source_improvement_pct",
                round(src_imp, 2)
            )
            mlflow.set_tag("stage", "finetuned")
            mlflow.set_tag("week", "3")

        print("\nResults logged to MLflow!")

    return ft_summary


if __name__ == "__main__":
    compare_answer_quality()