from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import os


def chunk_pages(pages: list, chunk_size: int = 512, 
                chunk_overlap: int = 50) -> list:
    """
    Split extracted pages into smaller overlapping chunks.
    Each chunk keeps track of its source document and page number.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    all_chunks = []
    chunk_id = 0
    
    for page in pages:
        # Skip empty pages
        if len(page["text"].strip()) < 10:
            continue
            
        # Split this page's text into chunks
        chunks = splitter.split_text(page["text"])
        
        for chunk_text in chunks:
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source": page["source"],
                "page": page["page"],
                "char_count": len(chunk_text)
            })
            chunk_id += 1
    
    return all_chunks


def save_chunks(chunks: list, output_path: str):
    """Save chunks to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    # Test with dummy text
    test_pages = [
        {
            "page": 1,
            "text": "Artificial intelligence is transforming industries. " * 50,
            "source": "test.pdf"
        }
    ]
    
    chunks = chunk_pages(test_pages)
    print(f"Input: 1 page of text")
    print(f"Output: {len(chunks)} chunks")
    print(f"First chunk ({chunks[0]['char_count']} chars):")
    print(chunks[0]["text"][:150])