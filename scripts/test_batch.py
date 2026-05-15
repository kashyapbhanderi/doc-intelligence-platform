import os
import sys
import json
import time
import logging
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('.'))

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def process_all(input_dir="data/raw", output_dir="data/processed"):
    from ingestion.merger import process_pdf_with_vision, save_document

    os.makedirs(output_dir, exist_ok=True)
    pdf_files = list(Path(input_dir).glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDFs to process")
    print("Processing WITHOUT vision (free, fast)...")
    print("=" * 50)

    results = []
    success = 0
    errors = 0
    total_chunks = 0
    start = time.time()

    for pdf_path in tqdm(pdf_files, desc="Processing"):
        # Skip already processed
        out_path = f"{output_dir}/{pdf_path.stem}_processed.json"
        if os.path.exists(out_path):
            with open(out_path) as f:
                existing = json.load(f)
            chunks = existing.get("total_chunks", 0)
            total_chunks += chunks
            success += 1
            results.append({
                "file": pdf_path.name,
                "status": "skipped",
                "chunks": chunks
            })
            continue

        try:
            doc = process_pdf_with_vision(
                str(pdf_path),
                use_vision=False,  # no API cost
                max_vision_pages=0
            )
            save_document(doc, output_dir)

            chunks = doc.get("total_chunks", 0)
            total_chunks += chunks
            success += 1

            results.append({
                "file": pdf_path.name,
                "status": "success",
                "pages": doc.get("total_pages", 0),
                "chunks": chunks
            })
            logger.info(f"OK: {pdf_path.name} — "
                       f"{doc.get('total_pages')} pages, "
                       f"{chunks} chunks")

        except Exception as e:
            errors += 1
            logger.error(f"FAILED: {pdf_path.name} — {e}")
            results.append({
                "file": pdf_path.name,
                "status": "error",
                "error": str(e),
                "chunks": 0
            })

    # Save summary
    summary = {
        "total_files": len(pdf_files),
        "successful": success,
        "errors": errors,
        "total_chunks": total_chunks,
        "time_seconds": round(time.time() - start, 1)
    }

    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    # Print final stats
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)
    print(f"Total PDFs:     {len(pdf_files)}")
    print(f"Successful:     {success}")
    print(f"Errors:         {errors}")
    print(f"Total chunks:   {total_chunks}")
    print(f"Avg per doc:    {total_chunks // max(success, 1)}")
    print(f"Time taken:     {summary['time_seconds']}s")
    print(f"Summary saved:  {output_dir}/summary.json")

    return summary


if __name__ == "__main__":
    process_all()