"""
eval/ragas_eval.py
RAGAS evaluation for your RAG pipeline.

4 metrics RAGAS measures:
──────────────────────────────────────────────────
1. Faithfulness
   Are claims in the answer supported by context?
   0 = hallucinated, 1 = fully grounded

2. Answer Relevancy
   Is the answer relevant to the question asked?
   0 = off-topic, 1 = perfectly on-topic

3. Context Recall
   Does retrieved context cover the ground truth?
   0 = retrieval missed key info, 1 = full coverage

4. Context Precision
   Is the retrieved context actually useful?
   0 = noisy retrieval, 1 = all context relevant
──────────────────────────────────────────────────

Industry use: Teams at Cohere, Anthropic, and
most RAG startups run RAGAS before every release
to catch quality regressions.
"""
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath('.'))

from dotenv import load_dotenv
load_dotenv()


def build_ragas_dataset(
    qa_path:  str = "eval/qa_dataset.json",
    limit:    int = 20
) -> list:
    """
    Build the evaluation dataset in RAGAS format.

    RAGAS needs:
    - question:   the user question
    - answer:     the generated answer from your RAG
    - contexts:   list of retrieved chunks used
    - ground_truth: the expected answer

    We generate answers by running each question
    through our actual RAG pipeline.
    """
    if not os.path.exists(qa_path):
        print(f"Q&A dataset not found: {qa_path}")
        print("Run: python eval/generate_qa.py")
        return []

    with open(qa_path, encoding='utf-8') as f:
        qa_pairs = json.load(f)

    qa_pairs = qa_pairs[:limit]
    print(f"Building RAGAS dataset from "
          f"{len(qa_pairs)} Q&A pairs...")

    # Import our pipeline
    from embeddings.query_engine import (
        build_query_engine,
        query_with_sources
    )
    from embeddings.embedder import DocumentEmbedder

    engine, _ = build_query_engine(top_k=5)

    dataset = []
    failed  = 0

    for i, qa in enumerate(qa_pairs):
        question = qa.get("question", "")
        ground_truth = qa.get("answer", "")

        if not question:
            continue

        print(f"  [{i+1:2}/{len(qa_pairs)}] "
              f"{question[:55]}...", end=" ")

        try:
            result = query_with_sources(
                engine, question)

            answer   = result.get("answer", "")
            sources  = result.get("sources", [])
            contexts = [
                s.get("text_preview", "")
                for s in sources
                if s.get("text_preview")
            ]

            if answer and contexts:
                dataset.append({
                    "question":    question,
                    "answer":      answer,
                    "contexts":    contexts,
                    "ground_truth": ground_truth,
                })
                print("✅")
            else:
                print("⚠️  empty answer/context")
                failed += 1

        except Exception as e:
            print(f"❌ {str(e)[:40]}")
            failed += 1

    print(f"\nDataset built: {len(dataset)} samples "
          f"({failed} failed)")
    return dataset


def run_ragas_evaluation(
    dataset:     list,
    output_path: str = "eval/ragas_results.json"
) -> dict | None:
    """
    Run RAGAS metrics on the dataset.

    Uses LLM-as-judge internally — requires
    OPENAI_API_KEY (or OpenRouter equivalent).

    Estimated cost: ~$0.05-0.10 for 20 samples.
    """
    if not dataset:
        print("Empty dataset — nothing to evaluate")
        return None

    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        from datasets import Dataset

        print(f"\nRunning RAGAS on "
              f"{len(dataset)} samples...")
        print("(Uses LLM-as-judge — takes ~3-5 min)")
        print("=" * 50)

        # Convert to HuggingFace Dataset
        hf_dataset = Dataset.from_dict({
            "question": [
                d["question"] for d in dataset],
            "answer": [
                d["answer"] for d in dataset],
            "contexts": [
                d["contexts"] for d in dataset],
            "ground_truth": [
                d["ground_truth"] for d in dataset],
        })

        # Configure OpenRouter if needed
        base_url = os.getenv(
            "OPENAI_BASE_URL",
            "https://api.openai.com/v1"
        )

        if "openrouter" in base_url:
            # RAGAS uses LangChain under the hood
            from langchain_openai import ChatOpenAI
            from ragas.llms import LangchainLLMWrapper

            llm = ChatOpenAI(
                model="llama-3.3-70b-versatile",
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=base_url,
                temperature=0,
                max_tokens=512
            )
            ragas_llm = LangchainLLMWrapper(llm)

            # Set LLM for all metrics
            for metric in [
                faithfulness, answer_relevancy,
                context_recall, context_precision
            ]:
                metric.llm = ragas_llm

        # Run evaluation
               # Run evaluation
        result = evaluate(
            dataset=hf_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            ]
        )

        import numpy as np

        # Modern RAGAS returns EvaluationResult
        result_df = result.to_pandas()

        scores = {
            "faithfulness": round(
                float(np.mean(result_df["faithfulness"])), 4),

            "answer_relevancy": round(
                float(np.mean(result_df["answer_relevancy"])), 4),

            "context_recall": round(
                float(np.mean(result_df["context_recall"])), 4),

            "context_precision": round(
                float(np.mean(result_df["context_precision"])), 4),

            "num_samples": len(dataset),
        }

        # Save results
        os.makedirs("eval", exist_ok=True)
        output = {
            "summary": scores,
            "model":   "all-MiniLM-L6-v2 (fine-tuned)",
            "stage":   "week-6"
        }
        with open(output_path, "w",
                  encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        return scores

    except ImportError as e:
        print(f"RAGAS import error: {e}")
        print("Run: pip install ragas datasets")
        return None
    except Exception as e:
        print(f"RAGAS evaluation error: {e}")
        return None


def print_ragas_summary(scores: dict):
    """Print a formatted RAGAS results table."""
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Samples evaluated: {scores['num_samples']}")
    print()

    metrics = [
        ("Faithfulness",
         scores["faithfulness"],
         "Are answers grounded in context?"),
        ("Answer Relevancy",
         scores["answer_relevancy"],
         "Do answers address the question?"),
        ("Context Recall",
         scores["context_recall"],
         "Is ground truth covered?"),
        ("Context Precision",
         scores["context_precision"],
         "Is retrieved context useful?"),
    ]

    for name, score, description in metrics:
        bar   = "█" * int(score * 20)
        grade = ("Excellent" if score >= 0.8
                 else "Good" if score >= 0.6
                 else "Fair" if score >= 0.4
                 else "Needs work")
        print(f"  {name:<22} {score:.4f}  "
              f"{bar:<20} {grade}")
        print(f"  {'':22} {description}")
        print()

    avg = sum(
        scores[k] for k in [
            "faithfulness", "answer_relevancy",
            "context_recall", "context_precision"
        ]
    ) / 4

    print(f"  {'Overall Average':<22} "
          f"{avg:.4f}")
    print("=" * 60)

    print("\n⭐ RESUME BULLET POINT:")
    print("-" * 60)
    print(
        f"Evaluated production RAG pipeline using RAGAS "
        f"framework: faithfulness={scores['faithfulness']:.2f}, "
        f"answer relevancy={scores['answer_relevancy']:.2f}, "
        f"context recall={scores['context_recall']:.2f}"
    )
    print("-" * 60)


if __name__ == "__main__":

    dataset = build_ragas_dataset(
        qa_path="eval/qa_dataset.json",
        limit=20
    )

    if not dataset:
        print("Could not build dataset. Exiting.")
        sys.exit(1)

    scores = run_ragas_evaluation(
        dataset,
        output_path="eval/ragas_results.json"
    )

    # Close Weaviate client properly
    try:
        from embeddings.query_engine import _weaviate_client

        if _weaviate_client:
            _weaviate_client.close()

    except Exception:
        pass

    # PRINT RESULTS
    if scores:
        print_ragas_summary(scores)
        print(
            "\nResults saved → "
            "eval/ragas_results.json"
        )

    else:
        print("RAGAS evaluation failed.")
        print("Check logs above.")