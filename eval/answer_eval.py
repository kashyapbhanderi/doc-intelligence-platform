import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath('.'))

from embeddings.query_engine import build_query_engine, query_with_sources


def keyword_overlap_score(answer: str, expected: str) -> float:
    """
    Fraction of expected-answer keywords that appear in the generated answer.
    Stop words are removed so common words don't inflate the score.
    This is word-level recall — a cheap proxy for answer quality
    that works without an extra API call.
    """
    answer_words   = set(answer.lower().split())
    expected_words = set(expected.lower().split())

    stop_words = {
        'the','a','an','is','are','was','were','be','been','being',
        'have','has','had','do','does','did','will','would','could',
        'should','may','might','must','can','to','of','in','on','at',
        'by','for','with','as','it','its','this','that','and','or'
    }

    expected_keywords = expected_words - stop_words
    if not expected_keywords:
        return 0.0

    matches = answer_words.intersection(expected_keywords)
    return len(matches) / len(expected_keywords)


def source_accuracy_score(sources: list, expected_source: str) -> float:
    """1.0 if the correct source paper appears in the top 3 results, else 0.0."""
    top_sources = [s.get("source", "") for s in sources[:3]]
    return 1.0 if expected_source in top_sources else 0.0


def evaluate_answers(
    qa_path="eval/qa_dataset.json",
    output_path="eval/answer_eval_results.json",
    limit=20
):
    if not os.path.exists(qa_path):
        print(f"Q&A dataset not found: {qa_path}")
        return

    with open(qa_path, encoding='utf-8') as f:
        qa_pairs = json.load(f)

    qa_pairs = qa_pairs[:limit]
    print(f"Evaluating {len(qa_pairs)} Q&A pairs")

    print("Building query engine...")
    engine, _ = build_query_engine(top_k=5)

    results        = []
    keyword_scores = []
    source_scores  = []
    latencies      = []

    print("\nRunning evaluation...")
    print("=" * 60)

    for i, qa in enumerate(qa_pairs):
        question        = qa.get("question", "")
        expected_answer = qa.get("answer", "")
        expected_source = qa.get("source", "")

        if not question:
            continue

        print(f"[{i+1:2}/{len(qa_pairs)}] {question[:50]}...", end=" ")

        start   = time.time()
        result  = query_with_sources(engine, question)
        elapsed = time.time() - start

        kw_score  = keyword_overlap_score(result["answer"], expected_answer)
        src_score = source_accuracy_score(result["sources"], expected_source)

        keyword_scores.append(kw_score)
        source_scores.append(src_score)
        latencies.append(elapsed)

        src_icon = "✅" if src_score == 1.0 else "❌"
        print(f"KW:{kw_score:.2f} Src:{src_icon} ({elapsed:.1f}s)")

        results.append({
            "question":          question,
            "expected_answer":   expected_answer[:200],
            "generated_answer":  result["answer"][:300],
            "expected_source":   expected_source,
            "retrieved_sources": [s["source"] for s in result["sources"]],
            "keyword_score":     round(kw_score, 4),
            "source_score":      src_score,
            "latency_seconds":   round(elapsed, 2)
        })

    avg_keyword = sum(keyword_scores) / len(keyword_scores)
    avg_source  = sum(source_scores) / len(source_scores)
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

    summary = {
        "total_evaluated":     len(results),
        "avg_keyword_overlap": round(avg_keyword, 4),
        "avg_source_accuracy": round(avg_source, 4),
        "avg_latency_seconds": round(avg_latency, 2),
        "p95_latency_seconds": round(p95_latency, 2),
        "model": "all-MiniLM-L6-v2",
        "stage": "baseline"
    }

    os.makedirs("eval", exist_ok=True)
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("ANSWER EVALUATION RESULTS")
    print("=" * 60)
    print(f"Questions evaluated:  {len(results)}")
    print(f"Keyword overlap:      {avg_keyword:.4f} ({'Good' if avg_keyword > 0.3 else 'Low'})")
    print(f"Source accuracy:      {avg_source:.4f} ({'Good' if avg_source > 0.5 else 'Low'})")
    print(f"Avg latency:          {avg_latency:.2f}s")
    print(f"P95 latency:          {p95_latency:.2f}s")
    print(f"\nResults saved: {output_path}")
    print("\n⭐ Save these baseline numbers!")
    print("   They improve after fine-tuning in Week 3")

    return summary


if __name__ == "__main__":
    evaluate_answers(limit=20)
    