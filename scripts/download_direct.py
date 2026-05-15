import os
import requests
import time

os.makedirs('data/raw', exist_ok=True)

# 50 famous AI/ML papers — direct download, no API needed
PAPERS = [
    # RAG and Retrieval
    ("https://arxiv.org/pdf/2005.11401", "rag_original.pdf"),
    ("https://arxiv.org/pdf/2208.09257", "rag_survey.pdf"),
    ("https://arxiv.org/pdf/2312.10997", "rag_advanced.pdf"),
    ("https://arxiv.org/pdf/2310.11511", "selfrag.pdf"),
    ("https://arxiv.org/pdf/2401.15884", "corrective_rag.pdf"),

    # LLMs
    ("https://arxiv.org/pdf/2302.13971", "llama.pdf"),
    ("https://arxiv.org/pdf/2307.09288", "llama2.pdf"),
    ("https://arxiv.org/pdf/2310.06825", "mistral.pdf"),
    ("https://arxiv.org/pdf/1706.03762", "attention.pdf"),
    ("https://arxiv.org/pdf/2301.07543", "chatgpt_survey.pdf"),

    # Fine-tuning
    ("https://arxiv.org/pdf/2106.09685", "lora.pdf"),
    ("https://arxiv.org/pdf/2305.10403", "qlora.pdf"),
    ("https://arxiv.org/pdf/2104.08691", "prefix_tuning.pdf"),
    ("https://arxiv.org/pdf/2110.07602", "finetuned_lm.pdf"),
    ("https://arxiv.org/pdf/2210.11610", "scaling_rlhf.pdf"),

    # Embeddings
    ("https://arxiv.org/pdf/1908.10084", "sentence_bert.pdf"),
    ("https://arxiv.org/pdf/2201.10005", "large_dual_encoder.pdf"),
    ("https://arxiv.org/pdf/2004.01630", "dense_retrieval.pdf"),
    ("https://arxiv.org/pdf/2112.09118", "cpt_text.pdf"),
    ("https://arxiv.org/pdf/2308.03281", "instructor.pdf"),

    # Agents
    ("https://arxiv.org/pdf/2210.03629", "react.pdf"),
    ("https://arxiv.org/pdf/2303.11366", "reflexion.pdf"),
    ("https://arxiv.org/pdf/2308.11432", "autogen.pdf"),
    ("https://arxiv.org/pdf/2305.10601", "tree_of_thought.pdf"),
    ("https://arxiv.org/pdf/2304.03442", "generative_agents.pdf"),

    # Document Understanding
    ("https://arxiv.org/pdf/2111.15664", "donut.pdf"),
    ("https://arxiv.org/pdf/2204.08387", "layoutlmv3.pdf"),
    ("https://arxiv.org/pdf/2212.09561", "dit.pdf"),
    ("https://arxiv.org/pdf/2309.10305", "nougat.pdf"),
    ("https://arxiv.org/pdf/2101.09465", "layoutlmv2.pdf"),

    # NLP Tasks
    ("https://arxiv.org/pdf/1810.04805", "bert.pdf"),
    ("https://arxiv.org/pdf/2005.14165", "gpt3.pdf"),
    ("https://arxiv.org/pdf/1910.10683", "t5.pdf"),
    ("https://arxiv.org/pdf/2109.01652", "flan.pdf"),
    ("https://arxiv.org/pdf/2203.02155", "chain_of_thought.pdf"),

    # Vector Search
    ("https://arxiv.org/pdf/2010.08241", "ann_benchmarks.pdf"),
    ("https://arxiv.org/pdf/1603.09320", "faiss.pdf"),
    ("https://arxiv.org/pdf/2112.07899", "beir.pdf"),
    ("https://arxiv.org/pdf/2204.07023", "mteb.pdf"),
    ("https://arxiv.org/pdf/2009.10195", "dpr.pdf"),

    # Evaluation
    ("https://arxiv.org/pdf/2306.05685", "ragas.pdf"),
    ("https://arxiv.org/pdf/2212.10560", "llm_eval.pdf"),
    ("https://arxiv.org/pdf/2304.15004", "g_eval.pdf"),
    ("https://arxiv.org/pdf/2307.03109", "shepherd.pdf"),
    ("https://arxiv.org/pdf/2310.01848", "prometheus.pdf"),

    # Multimodal
    ("https://arxiv.org/pdf/2304.08485", "minigpt4.pdf"),
    ("https://arxiv.org/pdf/2310.03744", "improved_llava.pdf"),
    ("https://arxiv.org/pdf/2305.11175", "instructblip.pdf"),
    ("https://arxiv.org/pdf/2111.01243", "flava.pdf"),
    ("https://arxiv.org/pdf/2103.00020", "clip.pdf"),
]


def download_pdf(url, save_path):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36"
        }
        r = requests.get(url, headers=headers, timeout=40)
        if r.status_code == 200 and len(r.content) > 5000:
            with open(save_path, "wb") as f:
                f.write(r.content)
            return True, os.path.getsize(save_path)
        else:
            return False, r.status_code
    except Exception as e:
        return False, str(e)


def main():
    print(f"Downloading {len(PAPERS)} AI/ML research papers...")
    print(f"Save location: data/raw/")
    print("=" * 50)

    success = 0
    failed = 0
    skipped = 0

    for i, (url, filename) in enumerate(PAPERS, 1):
        save_path = f"data/raw/{filename}"

        # Skip if already downloaded
        if os.path.exists(save_path):
            size = os.path.getsize(save_path)
            print(f"[{i:2}/{len(PAPERS)}] SKIP  {filename} "
                  f"({size // 1024} KB already exists)")
            skipped += 1
            continue

        # Download
        print(f"[{i:2}/{len(PAPERS)}] GET   {filename}...", end=" ")
        ok, info = download_pdf(url, save_path)

        if ok:
            print(f"OK ({info // 1024} KB)")
            success += 1
        else:
            print(f"FAILED ({info})")
            failed += 1

        # Wait between downloads to be polite
        time.sleep(2)

    # Final summary
    print("\n" + "=" * 50)
    print("DOWNLOAD COMPLETE")
    print("=" * 50)
    print(f"Downloaded:  {success}")
    print(f"Skipped:     {skipped}")
    print(f"Failed:      {failed}")
    print(f"Total files: {len(os.listdir('data/raw'))}")

    # Show all files
    print("\nFiles in data/raw/:")
    for f in sorted(os.listdir("data/raw")):
        size = os.path.getsize(f"data/raw/{f}")
        print(f"  {f} ({size // 1024} KB)")


if __name__ == "__main__":
    main()