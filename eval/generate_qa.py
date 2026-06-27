import os
import sys
import json
import time
import random
from pathlib import Path

sys.path.insert(0, os.path.abspath('.'))
from dotenv import load_dotenv
load_dotenv()


def load_random_chunks(processed_dir: str = "data/processed",
                       num_chunks: int = 60) -> list:
    """
    Load random chunks from processed documents.
    We pick random chunks to get diverse Q&A pairs
    covering all topics in our dataset.
    """
    all_chunks = []
    json_files = list(Path(processed_dir).glob(
        "*_processed.json"))

    for json_file in json_files:
        try:
            with open(json_file, encoding='utf-8') as f:
                doc = json.load(f)
            chunks = doc.get("chunks", [])
            # Only use chunks with enough text
            good_chunks = [
                c for c in chunks
                if len(c.get("text", "")) > 200
            ]
            all_chunks.extend(good_chunks)
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")

    # Shuffle and pick random chunks
    random.shuffle(all_chunks)
    selected = all_chunks[:num_chunks]
    print(f"Loaded {len(all_chunks)} total chunks")
    print(f"Selected {len(selected)} for Q&A generation")
    return selected


def generate_qa_from_chunk(chunk: dict,
                            client) -> dict | None:
    """
    Use GPT to generate one Q&A pair from a text chunk.

    The prompt asks for a question that:
    - Can ONLY be answered using this specific text
    - Is not too general or too specific
    - Tests real understanding

    Args:
        chunk: Text chunk dict
        client: OpenAI client

    Returns:
        Dict with question, answer, source, chunk_id
    """
    text = chunk.get("text", "")
    source = chunk.get("source", "")

    prompt = f"""Read this text from a research paper and 
create ONE good question-answer pair.

TEXT:
{text[:800]}

Rules:
- Question must be answerable ONLY from this text
- Answer must be a direct quote or close paraphrase
- Question should test real understanding
- Keep answer under 100 words

Return ONLY valid JSON like this:
{{
  "question": "your question here",
  "answer": "answer from the text here",
  "difficulty": "easy or medium or hard"
}}

Return ONLY the JSON. No explanation. No markdown."""

    try:
        from openai import OpenAI
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Groq fast inference
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )

        raw = response.choices[0].message.content.strip()

        # Clean markdown if present
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]

        qa = json.loads(raw.strip())
        qa["source"] = source
        qa["chunk_id"] = str(chunk.get("chunk_id", ""))
        qa["chunk_text"] = text[:500]
        qa["page"] = chunk.get("page", 0)
        return qa

    except json.JSONDecodeError:
        return None
    except Exception as e:
        print(f"  API error: {e}")
        return None


def generate_dataset(
    num_pairs: int = 50,
    output_path: str = "eval/qa_dataset.json",
    processed_dir: str = "data/processed"
):
    """
    Generate a complete Q&A evaluation dataset.

    Args:
        num_pairs: How many Q&A pairs to generate
        output_path: Where to save the dataset
        processed_dir: Where processed docs are stored
    """
    from openai import OpenAI

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env")
        print("Add: OPENAI_API_KEY=sk-your-key to .env file")
        return []

    base_url = os.getenv("OPENAI_BASE_URL","https://api.openai.com/v1")
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
        )
    os.makedirs("eval", exist_ok=True)

    # Load existing dataset if any
    if os.path.exists(output_path):
        with open(output_path, encoding='utf-8') as f:
            existing = json.load(f)
        print(f"Found {len(existing)} existing Q&A pairs")
        if len(existing) >= num_pairs:
            print(f"Already have {num_pairs} pairs. Done!")
            return existing
    else:
        existing = []

    needed = num_pairs - len(existing)
    print(f"Need to generate {needed} more Q&A pairs")

    # Load chunks to generate from
    chunks = load_random_chunks(
        processed_dir,
        num_chunks=needed + 20  # extra for failures
    )

    qa_pairs = existing.copy()
    attempts = 0
    failures = 0

    print(f"\nGenerating Q&A pairs...")
    print(f"Model: llama-3.3-70b-versatile (Groq fast inference)")
    print(f"Estimated cost: ~${needed * 0.001:.3f}")
    print("-" * 50)

    for chunk in chunks:
        if len(qa_pairs) >= num_pairs:
            break

        attempts += 1
        current = len(qa_pairs) + 1
        print(f"[{current:2}/{num_pairs}] "
              f"Source: {chunk.get('source', '')[:30]}...",
              end=" ")

        qa = generate_qa_from_chunk(chunk, client)

        if qa and qa.get("question") and qa.get("answer"):
            qa_pairs.append(qa)
            print(f"✅ {qa['difficulty']}")
        else:
            failures += 1
            print("❌ failed")

        # Save after every 5 pairs
        if len(qa_pairs) % 5 == 0:
            with open(output_path, "w",
                      encoding='utf-8') as f:
                json.dump(qa_pairs, f, indent=2,
                          ensure_ascii=False)

        # Rate limit delay
        time.sleep(0.5)

    # Final save
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2,
                  ensure_ascii=False)

    print("\n" + "=" * 50)
    print("Q&A GENERATION COMPLETE")
    print("=" * 50)
    print(f"Total pairs:  {len(qa_pairs)}")
    print(f"Attempts:     {attempts}")
    print(f"Failures:     {failures}")
    print(f"Saved to:     {output_path}")

    # Show difficulty breakdown
    easy = len([q for q in qa_pairs
                if q.get("difficulty") == "easy"])
    medium = len([q for q in qa_pairs
                  if q.get("difficulty") == "medium"])
    hard = len([q for q in qa_pairs
                if q.get("difficulty") == "hard"])
    print(f"\nDifficulty breakdown:")
    print(f"  Easy:   {easy}")
    print(f"  Medium: {medium}")
    print(f"  Hard:   {hard}")

    return qa_pairs


if __name__ == "__main__":
    pairs = generate_dataset(
        num_pairs=60,
        output_path="eval/qa_dataset.json"
    )
    print(f"\nSample Q&A pair:")
    if pairs:
        sample = pairs[0]
        print(f"Q: {sample['question']}")
        print(f"A: {sample['answer'][:150]}")
        print(f"Source: {sample['source']}")