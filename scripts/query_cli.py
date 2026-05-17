import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath('.'))

from embeddings.query_engine import build_query_engine, query_with_sources
from embeddings.embedder import DocumentEmbedder


def format_answer(result: dict) -> str:
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append(f"QUESTION: {result['question']}")
    lines.append("=" * 60)
    lines.append(f"\nANSWER:\n{result['answer']}")
    lines.append(f"\nSOURCES ({result['num_sources']}):")
    for i, src in enumerate(result['sources'][:5], 1):
        lines.append(f"  {i}. {src['source']} (page {src['page']}, score: {src['score']})")
        lines.append(f"     Preview: {src['text_preview'][:100]}...")
    lines.append("=" * 60)
    return "\n".join(lines)


def run_test_questions(engine):
    test_questions = [
        "What is retrieval augmented generation?",
        "How does LoRA fine-tuning reduce memory usage?",
        "What is the attention mechanism in transformers?",
        "How does BERT differ from GPT?",
        "What is contrastive learning for embeddings?",
        "How do agents use tools in LLM systems?",
        "What is chain of thought prompting?",
        "How does RLHF improve language models?",
        "What is the difference between dense and sparse retrieval?",
        "How do sentence transformers create embeddings?",
        "Your question: What is the difference between LoRA and QLoRA?"
    ]

    print("\nRunning 10 test questions...")
    print("=" * 60)

    results = []
    for i, question in enumerate(test_questions, 1):
        print(f"\n[{i:2}/10] {question}")
        print("-" * 50)

        start  = time.time()
        result = query_with_sources(engine, question)
        elapsed = time.time() - start

        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources: {result['num_sources']} | Time: {elapsed:.1f}s")

        results.append({**result, "latency_seconds": round(elapsed, 2)})

    os.makedirs("eval", exist_ok=True)
    with open("eval/test_questions_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    avg_latency = sum(r["latency_seconds"] for r in results) / len(results)
    avg_sources = sum(r["num_sources"] for r in results) / len(results)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Questions answered: {len(results)}/10")
    print(f"Avg latency:        {avg_latency:.1f}s")
    print(f"Avg sources used:   {avg_sources:.1f}")
    print(f"Results saved:      eval/test_questions_results.json")
    return results


def interactive_mode(engine):
    print("\n" + "=" * 60)
    print("INTERACTIVE Q&A MODE")
    print("Ask questions about your 52 AI/ML papers")
    print("Type 'quit' to exit | 'test' to run test questions")
    print("=" * 60)

    while True:
        try:
            print()
            question = input("Your question: ").strip()

            if not question:
                continue
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if question.lower() == 'test':
                run_test_questions(engine)
                continue

            start   = time.time()
            result  = query_with_sources(engine, question)
            elapsed = time.time() - start

            print(format_answer(result))
            print(f"Response time: {elapsed:.1f}s")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    print("Loading query engine...")
    print("(This takes ~20 seconds first time)\n")

    engine, client = build_query_engine(top_k=5)

    embedder = DocumentEmbedder()
    count    = embedder.get_document_count()
    print(f"Documents in database: {count} chunks")
    print(f"Source papers: 52 AI/ML research papers")

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_test_questions(engine)
    else:
        interactive_mode(engine)