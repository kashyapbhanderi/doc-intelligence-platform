"""
eval/agent_eval.py
Full pipeline evaluation — 20 questions through
the complete Planner → Executor → Critic loop.

Measures:
- Faithfulness rate (% answers approved by Critic)
- Source accuracy (correct paper retrieved)
- Avg latency per question
- Answer length
"""
import os
import sys
import json
import time
sys.path.insert(0, os.path.abspath('.'))

import mlflow
from agents.graph import ask


# 20 diverse test questions
TEST_QUESTIONS = [
    "What is retrieval augmented generation?",
    "How does LoRA reduce memory during fine-tuning?",
    "What is the attention mechanism in transformers?",
    "How does BERT differ from GPT architecturally?",
    "What is contrastive learning for embeddings?",
    "How do agents use tools in LLM systems?",
    "What is chain of thought prompting?",
    "How does RLHF improve language model alignment?",
    "What is the difference between dense and sparse retrieval?",
    "How do sentence transformers create embeddings?",
    "What problem does layer normalisation solve?",
    "How does self-attention scale with sequence length?",
    "What is the purpose of the critic in RAG systems?",
    "How does hybrid search combine BM25 and vectors?",
    "What is reciprocal rank fusion?",
    "How does QLoRA differ from LoRA?",
    "What is the role of the embedding model in RAG?",
    "How does query decomposition improve retrieval?",
    "What metrics measure RAG system quality?",
    "How does fine-tuning improve embedding models?",
]

OUTPUT_PATH = "eval/agent_eval_results.json"


def run_agent_evaluation():
    """
    Run all 20 questions through the full agent pipeline.
    Log results to MLflow and save to JSON.
    """
    print("=" * 60)
    print("FULL PIPELINE EVALUATION")
    print(f"{len(TEST_QUESTIONS)} questions × "
          f"3 agents (Planner → Executor → Critic)")
    print("=" * 60)

    results    = []
    latencies  = []
    faithful   = 0
    total_srcs = 0

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i:2}/{len(TEST_QUESTIONS)}] "
              f"{question[:55]}...")

        start  = time.time()
        result = ask(question, verbose=False)
        elapsed = time.time() - start

        latencies.append(elapsed)
        is_faithful = result.get("is_faithful", False)
        if is_faithful:
            faithful += 1

        n_sources = len(result.get("sources", []))
        total_srcs += n_sources

        # Print one-line summary
        faith_icon = "✅" if is_faithful else "❌"
        print(f"   {faith_icon} faithful={is_faithful} | "
              f"sources={n_sources} | "
              f"time={elapsed:.1f}s | "
              f"sub_q={len(result.get('sub_queries',[]))}")

        results.append({
            "question":    question,
            "sub_queries": result.get("sub_queries", []),
            "answer":      result.get(
                "final_answer", "")[:300],
            "is_faithful": is_faithful,
            "critique":    result.get("critique", ""),
            "num_sources": n_sources,
            "sources": [
                s["source"]
                for s in result.get("sources", [])[:3]
            ],
            "latency_seconds": round(elapsed, 2),
        })

    # Calculate summary stats
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[
        int(len(latencies) * 0.95)]
    faithfulness_rate = faithful / len(TEST_QUESTIONS)
    avg_sources = total_srcs / len(TEST_QUESTIONS)

    summary = {
        "total_questions":    len(TEST_QUESTIONS),
        "faithful_count":     faithful,
        "faithfulness_rate":  round(faithfulness_rate, 4),
        "avg_latency":        round(avg_latency, 2),
        "p95_latency":        round(p95_latency, 2),
        "avg_sources":        round(avg_sources, 2),
    }

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Questions answered:  {len(TEST_QUESTIONS)}")
    print(f"Faithfulness rate:   "
          f"{faithfulness_rate:.1%}  "
          f"({faithful}/{len(TEST_QUESTIONS)})")
    print(f"Avg latency:         {avg_latency:.1f}s")
    print(f"P95 latency:         {p95_latency:.1f}s")
    print(f"Avg sources/answer:  {avg_sources:.1f}")
    print("=" * 60)

    # Log to MLflow
    mlflow.set_experiment("agent-evaluation")
    with mlflow.start_run(
        run_name="full-pipeline-20q"
    ):
        mlflow.log_param("num_questions",
                         len(TEST_QUESTIONS))
        mlflow.log_param("agents",
                         "planner-executor-critic")
        mlflow.log_metric("faithfulness_rate",
                          faithfulness_rate)
        mlflow.log_metric("avg_latency",
                          avg_latency)
        mlflow.log_metric("p95_latency",
                          p95_latency)
        mlflow.log_metric("avg_sources",
                          avg_sources)
        mlflow.set_tag("week", "4")
        mlflow.set_tag("stage", "integration-test")

    # Save results
    os.makedirs("eval", exist_ok=True)
    with open(OUTPUT_PATH, "w",
               encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "results": results},
            f, indent=2, ensure_ascii=False
        )

    print(f"\nResults saved → {OUTPUT_PATH}")
    print("Logged to MLflow experiment: agent-evaluation")
    print("\n⭐ Write these numbers in your README!")
    return summary


if __name__ == "__main__":
    run_agent_evaluation()