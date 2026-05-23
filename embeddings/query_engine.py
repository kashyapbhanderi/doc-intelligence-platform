# import os
# import sys
# from dotenv import load_dotenv

# load_dotenv()
# sys.path.insert(0, os.path.abspath('.'))

# from sentence_transformers import SentenceTransformer
# import weaviate
# from openai import OpenAI as OpenAIClient

# weaviate_url = "http://localhost:8080"

# EMBED_MODEL = "all-MiniLM-L6-v2"
# COLLECTION  = "Document"

# _embedder       = None
# _weaviate_client = None

# def get_llm():
#     from llama_index.llms.openai import OpenAI as LlamaOpenAI

#     kwargs = {
#         "model": "gpt-4o-mini",
#         "temperature": 0.1,
#         "api_key": os.getenv("OPENAI_API_KEY"),
#     }
#     base_url = os.getenv("OPENAI_BASE_URL")
#     if base_url:
#         kwargs["api_base"] = base_url
#     return LlamaOpenAI(**kwargs)


# def get_embedder():
#     global _embedder
#     if _embedder is None:
#         print("Loading embedding model...")
#         _embedder = SentenceTransformer(EMBED_MODEL)
#     return _embedder


# def get_weaviate_client(url="http://localhost:8080"):
#     global _weaviate_client
#     if _weaviate_client is None:
#         _weaviate_client = weaviate.Client(url)
#     return _weaviate_client


# def retrieve_chunks(question, top_k=5, url=None) -> list:
#     """Embed question and search Weaviate for the most relevant chunks."""
#     embedder = get_embedder()
#     client   = get_weaviate_client(url or weaviate_url)

#     query_vector = embedder.encode(question).tolist()

#     raw = (
#         client.query
#         .get(COLLECTION, ["text", "source", "page"])
#         .with_near_vector({"vector": query_vector})
#         .with_additional(["distance"])
#         .with_limit(top_k)
#         .do()
#     )

#     docs = raw.get("data", {}).get("Get", {}).get(COLLECTION, []) or []

#     chunks = []
#     for doc in docs:
#         distance = float(doc.get("_additional", {}).get("distance", 1.0))
#         chunks.append({
#             "text":     doc.get("text", ""),
#             "source":   doc.get("source", ""),
#             "page":     doc.get("page", 0),
#             # "chunk_id": doc.get("chunk_id", ""),
#             "score":    round(1.0 - distance, 4),
#         })

#     return chunks


# def generate_answer(question: str, chunks: list) -> str:
#     """Build a grounded prompt from retrieved chunks and call the LLM."""
#     api_key  = os.getenv("OPENAI_API_KEY")
#     base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

#     client = OpenAIClient(
#         api_key=api_key,
#         base_url=base_url,
#         default_headers={
#             "HTTP-Referer": "https://github.com/kashyapbhanderi/doc-intelligence-platform",
#             "X-Title": "Doc Intelligence Platform"
#         }
#     )

#     context = "\n\n".join(
#         f"[{i}] Source: {c['source']} (page {c['page']})\n{c['text']}"
#         for i, c in enumerate(chunks[:5], 1)
#     )

#     prompt = f"""Answer the question using ONLY the context below.
# If the context does not contain enough information, say so clearly.
# Always mention which source you used.

# Context:
# {context}

# Question: {question}

# Answer:"""

#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.1,
#         max_tokens=512
#     )

#     return response.choices[0].message.content


# def build_query_engine(
#     weaviate_url: str = "http://localhost:8080",
#     top_k: int = 5,
#     model_path: str = None
# ):
#     """
#     Build a LlamaIndex query engine connected to Weaviate.

#     Now supports both base and fine-tuned models.

#     Args:
#         weaviate_url: Weaviate connection URL
#         top_k: Number of chunks to retrieve
#         model_path: Path to fine-tuned model.
#                     If None, uses base model.

#     Returns:
#         query engine, weaviate client
#     """
#     import weaviate
#     from llama_index.core import VectorStoreIndex, Settings
#     from llama_index.vector_stores.weaviate import (
#         WeaviateVectorStore
#     )
#     from llama_index.core.retrievers import (
#         VectorIndexRetriever
#     )
#     from llama_index.core.query_engine import (
#         RetrieverQueryEngine
#     )
#     from llama_index.core.postprocessor import (
#         SimilarityPostprocessor
#     )
#     from llama_index.core.embeddings import (
#         BaseEmbedding
#     )
#     from sentence_transformers import SentenceTransformer
#     import numpy as np

#     # Custom embedding class for LlamaIndex
#     class CustomEmbedding(BaseEmbedding):
#         """Wraps sentence-transformers for LlamaIndex."""

#         def __init__(self, model_path_or_name: str):
#             super().__init__()
#             self._st_model = SentenceTransformer(
#                 model_path_or_name)
#             model_name = (
#                 model_path_or_name
#                 if isinstance(model_path_or_name, str)
#                 else "custom"
#             )
#             self.model_name = model_name

#         def _get_query_embedding(
#             self, query: str
#         ) -> list:
#             return self._st_model.encode(
#                 query).tolist()

#         def _get_text_embedding(
#             self, text: str
#         ) -> list:
#             return self._st_model.encode(
#                 text).tolist()

#         def _get_text_embeddings(
#             self, texts: list
#         ) -> list:
#             return self._st_model.encode(
#                 texts).tolist()

#         async def _aget_query_embedding(
#             self, query: str
#         ) -> list:
#             return self._get_query_embedding(query)

#         async def _aget_text_embedding(
#             self, text: str
#         ) -> list:
#             return self._get_text_embedding(text)

#     # Determine which model to use
#     ft_path = "models/finetuned/final"
#     if model_path and os.path.exists(model_path):
#         use_model = model_path
#         print(f"Using provided model: {model_path}")
#     elif os.path.exists(ft_path):
#         use_model = ft_path
#         print(f"Using fine-tuned model: {ft_path}")
#     else:
#         use_model = "all-MiniLM-L6-v2"
#         print(f"Using base model: {use_model}")

#     # Setup embedding model
#     embed_model = CustomEmbedding(use_model)
#     Settings.embed_model = embed_model

#     # Connect to Weaviate
#     print(f"Connecting to Weaviate: {weaviate_url}")
#     client = weaviate.Client(weaviate_url)

#     # Build index
#     print("Building vector store index...")
#     vector_store = WeaviateVectorStore(
#         weaviate_client=client,
#         index_name="Document",
#         text_key="text"
#     )

#     index = VectorStoreIndex.from_vector_store(
#         vector_store=vector_store
#     )

#     Settings.llm = get_llm()

#     retriever = VectorIndexRetriever(
#         index=index,
#         similarity_top_k=top_k
#     )

#     query_engine = RetrieverQueryEngine(
#         retriever=retriever,
#         node_postprocessors=[
#             SimilarityPostprocessor(
#                 similarity_cutoff=0.3
#             )
#         ]
#     )
#     query_engine.top_k = top_k
#     query_engine.weaviate_url = weaviate_url

#     print("Query engine ready!")
#     return query_engine, client


# def query_with_sources(query_engine, question: str) -> dict:
#     """Ask a question and return answer + source citations."""
#     chunks = retrieve_chunks(
#         question,
#         top_k=getattr(query_engine, "top_k", 5),
#         url=getattr(query_engine, "weaviate_url", weaviate_url),
#     )

#     if not chunks:
#         return {"question": question, "answer": "No relevant documents found.",
#                 "sources": [], "num_sources": 0}

#     answer  = generate_answer(question, chunks)
#     sources = [{"source": c["source"], "page": c["page"],
#                 "score": c["score"], "text_preview": c["text"][:150]}
#                for c in chunks]

#     return {"question": question, "answer": answer,
#             "sources": sources, "num_sources": len(sources)}


# if __name__ == "__main__":
#     print("Building query engine...")
#     engine, client = build_query_engine(top_k=5)

#     test_q = "What is retrieval augmented generation?"
#     print(f"\nTest question: {test_q}")
#     print("-" * 50)

#     result = query_with_sources(engine, test_q)
#     print(f"Answer: {result['answer'][:300]}")
#     print(f"\nSources used: {result['num_sources']}")
#     for s in result['sources'][:3]:
#         print(f"  - {s['source']} (page {s['page']}, score: {s['score']})")



import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.abspath('.'))

from sentence_transformers import SentenceTransformer
import weaviate
from openai import OpenAI as OpenAIClient

weaviate_url = "http://localhost:8080"

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION  = "Document"

_embedder       = None
_weaviate_client = None

def get_llm():
    from llama_index.llms.openai import OpenAI as LlamaOpenAI

    kwargs = {
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "logprobs": False,
        "default_headers": {},
    }
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["api_base"] = base_url
        kwargs["default_headers"] = {
            "HTTP-Referer": "https://github.com/kashyapbhanderi/doc-intelligence-platform",
            "X-Title": "Doc Intelligence Platform",
        }
    return LlamaOpenAI(**kwargs)


def get_embedder():
    global _embedder
    if _embedder is None:
        print("Loading embedding model...")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_weaviate_client(url="http://localhost:8080"):
    global _weaviate_client
    if _weaviate_client is None:
        _weaviate_client = weaviate.connect_to_local(
            host="weaviate",
            port=8080,
            grpc_port=50051,
            skip_init_checks=True
        )
    return _weaviate_client


def retrieve_chunks(question, top_k=5, url=None) -> list:
    """Embed question and search Weaviate for the most relevant chunks."""
    embedder = get_embedder()
    client   = get_weaviate_client(url or weaviate_url)

    query_vector = embedder.encode(question).tolist()

    raw = (
        client.query
        .get(COLLECTION, ["text", "source", "page"])
        .with_near_vector({"vector": query_vector})
        .with_additional(["distance"])
        .with_limit(top_k)
        .do()
    )

    docs = raw.get("data", {}).get("Get", {}).get(COLLECTION, []) or []

    chunks = []
    for doc in docs:
        distance = float(doc.get("_additional", {}).get("distance", 1.0))
        chunks.append({
            "text":     doc.get("text", ""),
            "source":   doc.get("source", ""),
            "page":     doc.get("page", 0),
            # "chunk_id": doc.get("chunk_id", ""),
            "score":    round(1.0 - distance, 4),
        })

    return chunks


def generate_answer(question: str, chunks: list) -> str:
    """Build a grounded prompt from retrieved chunks and call the LLM."""
    api_key  = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    client = OpenAIClient(
        api_key=api_key,
        base_url=base_url,
        default_headers={
            "HTTP-Referer": "https://github.com/kashyapbhanderi/doc-intelligence-platform",
            "X-Title": "Doc Intelligence Platform"
        }
    )

    context = "\n\n".join(
        f"[{i}] Source: {c['source']} (page {c['page']})\n{c['text']}"
        for i, c in enumerate(chunks[:5], 1)
    )

    prompt = f"""Answer the question using ONLY the context below.
If the context does not contain enough information, say so clearly.
Always mention which source you used.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512
    )

    return response.choices[0].message.content


def build_query_engine(
    weaviate_url: str = "http://localhost:8080",
    top_k: int = 5,
    model_path: str = None
):
    """
    Build a LlamaIndex query engine connected to Weaviate.

    Now supports both base and fine-tuned models.

    Args:
        weaviate_url: Weaviate connection URL
        top_k: Number of chunks to retrieve
        model_path: Path to fine-tuned model.
                    If None, uses base model.

    Returns:
        query engine, weaviate client
    """
    import weaviate
    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.vector_stores.weaviate import (
        WeaviateVectorStore
    )
    from llama_index.core.retrievers import (
        VectorIndexRetriever
    )
    from llama_index.core.query_engine import (
        RetrieverQueryEngine
    )
    from llama_index.core.postprocessor import (
        SimilarityPostprocessor
    )
    from llama_index.core.embeddings import (
        BaseEmbedding
    )
    from sentence_transformers import SentenceTransformer
    import numpy as np

    # Custom embedding class for LlamaIndex
    class CustomEmbedding(BaseEmbedding):
        """Wraps sentence-transformers for LlamaIndex."""

        def __init__(self, model_path_or_name: str):
            super().__init__()
            self._st_model = SentenceTransformer(
                model_path_or_name)
            model_name = (
                model_path_or_name
                if isinstance(model_path_or_name, str)
                else "custom"
            )
            self.model_name = model_name

        def _get_query_embedding(
            self, query: str
        ) -> list:
            return self._st_model.encode(
                query).tolist()

        def _get_text_embedding(
            self, text: str
        ) -> list:
            return self._st_model.encode(
                text).tolist()

        def _get_text_embeddings(
            self, texts: list
        ) -> list:
            return self._st_model.encode(
                texts).tolist()

        async def _aget_query_embedding(
            self, query: str
        ) -> list:
            return self._get_query_embedding(query)

        async def _aget_text_embedding(
            self, text: str
        ) -> list:
            return self._get_text_embedding(text)

    # Determine which model to use
    ft_path = "models/finetuned/final"
    if model_path and os.path.exists(model_path):
        use_model = model_path
        print(f"Using provided model: {model_path}")
    elif os.path.exists(ft_path):
        use_model = ft_path
        print(f"Using fine-tuned model: {ft_path}")
    else:
        use_model = "all-MiniLM-L6-v2"
        print(f"Using base model: {use_model}")

    # Setup embedding model
    embed_model = CustomEmbedding(use_model)
    Settings.embed_model = embed_model

    # Connect to Weaviate
    print(f"Connecting to Weaviate: {weaviate_url}")
    client = client = weaviate.connect_to_local(
            host="weaviate",
            port=8080,
            grpc_port=50051,
            skip_init_checks=True
        )

    # Build index
    print("Building vector store index...")
    vector_store = WeaviateVectorStore(
        weaviate_client=client,
        index_name="Document",
        text_key="text"
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store
    )

    Settings.llm = get_llm()

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[
            SimilarityPostprocessor(
                similarity_cutoff=0.3
            )
        ]
    )
    query_engine.top_k = top_k
    query_engine.weaviate_url = weaviate_url

    print("Query engine ready!")
    return query_engine, client


def query_with_sources(query_engine, question: str) -> dict:
    """Ask a question and return answer + source citations."""
    chunks = retrieve_chunks(
        question,
        top_k=getattr(query_engine, "top_k", 5),
        url=getattr(query_engine, "weaviate_url", weaviate_url),
    )

    if not chunks:
        return {"question": question, "answer": "No relevant documents found.",
                "sources": [], "num_sources": 0}

    answer  = generate_answer(question, chunks)
    sources = [{"source": c["source"], "page": c["page"],
                "score": c["score"], "text_preview": c["text"][:150]}
               for c in chunks]

    return {"question": question, "answer": answer,
            "sources": sources, "num_sources": len(sources)}


if __name__ == "__main__":
    print("Building query engine...")
    engine, client = build_query_engine(top_k=5)

    test_q = "What is retrieval augmented generation?"
    print(f"\nTest question: {test_q}")
    print("-" * 50)

    result = query_with_sources(engine, test_q)
    print(f"Answer: {result['answer'][:300]}")
    print(f"\nSources used: {result['num_sources']}")
    for s in result['sources'][:3]:
        print(f"  - {s['source']} (page {s['page']}, score: {s['score']})")