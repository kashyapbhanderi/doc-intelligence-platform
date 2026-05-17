import os
import sys
import json
sys.path.insert(0, os.path.abspath('.'))


def clean_qa_dataset(
    input_path="eval/qa_dataset.json",
    output_path="eval/qa_dataset.json",
    min_question_len=15,
    min_answer_len=20
):
    """
    Remove poor quality Q&A pairs from dataset.
    Filters out pairs that are too short or look wrong.
    """
    with open(input_path, encoding='utf-8') as f:
        pairs = json.load(f)

    print(f"Before cleaning: {len(pairs)} pairs")

    clean_pairs = []
    removed = 0

    for pair in pairs:
        question = pair.get("question", "")
        answer = pair.get("answer", "")
        source = pair.get("source", "")

        # Skip if question too short
        if len(question) < min_question_len:
            print(f"  Removed short question: '{question}'")
            removed += 1
            continue

        # Skip if answer too short
        if len(answer) < min_answer_len:
            print(f"  Removed short answer: '{answer}'")
            removed += 1
            continue

        # Skip if question is just numbers
        if question.strip().isdigit():
            print(f"  Removed numeric question: '{question}'")
            removed += 1
            continue

        # Skip if no source
        if not source:
            removed += 1
            continue

        clean_pairs.append(pair)

    print(f"After cleaning:  {len(clean_pairs)} pairs")
    print(f"Removed:         {removed} bad pairs")

    # Save cleaned dataset
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(clean_pairs, f, indent=2,
                  ensure_ascii=False)

    print(f"Saved to: {output_path}")
    return clean_pairs


if __name__ == "__main__":
    clean_qa_dataset()