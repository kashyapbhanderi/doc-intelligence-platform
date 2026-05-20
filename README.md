# Doc Intelligence Platform

> Production-grade Multimodal Agentic RAG System with Document Editing Agent

![CI](https://github.com/kashyapbhanderi/doc-intelligence-platform/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Weaviate](https://img.shields.io/badge/Weaviate-1.24-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![Tests](https://img.shields.io/badge/Tests-31%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/Tests-78%20passing-brightgreen)

---

## What This Project Does

An end-to-end AI system that:

- Ingests PDF, image, and Word documents (text + scanned)
- Understands charts and tables using GPT-4o Vision
- Answers questions using a 3-agent RAG pipeline
- Edits `.docx`, `.pdf`, and image files on command
- Deployed on AWS with live monitoring dashboards

---

## Architecture
┌─────────────────────────────────────────────────────┐
│                   USER REQUEST                       │
│          (question or edit instruction)              │
└──────────────────────┬──────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────┐
│                 PLANNER AGENT                        │
│     Decomposes task into sub-queries                 │
│     Routes to RAG branch or Editor branch            │
└────────────┬─────────────────────┬──────────────────┘
│                     │
▼                     ▼
┌────────────────────┐  ┌─────────────────────────────┐
│   EXECUTOR AGENT   │  │      EDITOR AGENT            │
│  Hybrid Search     │  │  python-docx / PyMuPDF       │
│  (BM25 + Vector)   │  │  Pillow / rembg              │
│  Weaviate DB       │  │  pdf2docx converter          │
└────────────┬───────┘  └─────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────┐
│                  CRITIC AGENT                        │
│     Checks answer faithfulness vs context            │
│     Flags hallucinations before returning            │
└──────────────────────┬──────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────┐
│              VERIFIED ANSWER / EDITED FILE           │
└─────────────────────────────────────────────────────┘

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Ingestion | PyMuPDF | PDF text extraction |
| Ingestion | PaddleOCR | Scanned PDF fallback |
| Ingestion | GPT-4o Vision | Chart and table understanding |
| Embeddings | sentence-transformers | Text to vector conversion |
| Embeddings | LoRA / PEFT | Domain-specific fine-tuning |
| Vector DB | Weaviate | Store and search embeddings |
| Search | BM25 + RRF | Keyword search |
| Search | Dense vectors | Semantic search |
| Search | Hybrid RRF | Combined best results |
| Agents | LangGraph | Multi-agent orchestration |
| Editor | python-docx | Word document editing |
| Editor | PyMuPDF | PDF operations |
| Editor | Pillow + rembg | Image processing |
| API | FastAPI | REST API server |
| Deploy | Docker | Containerization |
| Deploy | AWS EC2 | Cloud hosting |
| MLOps | MLflow | Experiment tracking |
| Eval | RAGAS | RAG quality measurement |
| Monitor | Prometheus + Grafana | Live dashboards |

---

## Results

### Week 1 — Data Pipeline ✅

| Metric | Value |
|--------|-------|
| PDFs downloaded | 52 |
| PDFs processed | 52 / 52 (0 errors) |
| Total chunks created | 11,089 |
| Avg chunks per document | 213 |
| Largest document | llama2.pdf (611 chunks) |
| Supports scanned PDFs | ✅ OCR fallback |
| Supports charts/tables | ✅ GPT-4o Vision |
| Tests passing | 17 / 17 |
| CI pipeline | ✅ Green |

### Week 2 — Complete ✅

| Metric | Value |
|--------|-------|
| Embedding model | all-MiniLM-L6-v2 (384-dim) |
| Chunks in Weaviate | 11,089 |
| Q&A evaluation pairs | 44 (cleaned) |
| Hybrid NDCG@10 baseline | YOUR_SCORE |
| Source accuracy | YOUR_SCORE |
| Avg query latency | YOUR_SECONDs |
| Fine-tuning triplets | 274 (validated) |
| Tests passing | 59/59 |

## Model Weights
Fine-tuned model checkpoints are not stored in this repo (too large for GitHub).

### Week 3 — Fine-tuning ✅

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | all-MiniLM-L6-v2 |
| Training triplets | 274 (validated) |
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Loss function | MultipleNegativesRankingLoss |

#### NDCG@10 Results

| Method | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| BM25 | YOUR_SCORE | YOUR_SCORE | +X% |
| Vector | YOUR_SCORE | YOUR_SCORE | +X% |
| **Hybrid** | **YOUR_SCORE** | **YOUR_SCORE** | **+X%** |

#### Answer Quality

| Metric | Baseline | Fine-tuned | Change |
|--------|----------|------------|--------|
| Source accuracy | YOUR | YOUR | +X% |
| Keyword overlap | YOUR | YOUR | +X% |
| Avg latency | Xs | Xs | - |

> Fine-tuned `all-MiniLM-L6-v2` on 274 domain-specific
> AI/ML triplets using contrastive learning,
> achieving +X% improvement in retrieval NDCG@10.

### Search Method Comparison

| Method | Correct | Score | Best for |
|--------|---------|-------|---------|
| BM25 (keyword) | 3/5 | 60% | Exact terms |
| Vector (semantic) | 3/5 | 60% | Meaning-based |
| Hybrid RRF | 5/5 | 100% | Everything |

> **Hybrid RRF chosen as default** — combines BM25 keyword
> matching with vector semantic search using Reciprocal Rank
> Fusion algorithm for best overall performance.

![baseline-answer-quality](image.png)
![retrieval-quality(baseline-all-MiniLM)](<Screenshot 2026-05-18 104743.png>)
---

### Week 4 — Multi-Agent RAG ✅

| Metric | Value |
|--------|-------|
| Architecture | Planner → Executor → Critic |
| Questions tested | 20 |
| Faithfulness rate | YOUR_RATE% |
| Avg query latency | YOUR_SECONDs |
| P95 latency | YOUR_SECONDs |
| Avg sources/answer | YOUR_COUNT |
| Tests passing | 136/136 |

## Project Progress

| Week | Topic | Status | Tests |
|------|-------|--------|-------|
| Week 1 | Data Pipeline + OCR + Vision | ✅ Complete | 17/17 |
| Week 2 | Embeddings + Vector DB + RAG | ✅ Complete | 59/59 |
| Week 3 | Fine-tuning | ✅ Complete | 78/78 |
| Week 4 | Multi-Agent RAG | ✅ Complete | 136/136 |
| Week 5 | Document Editor Agent | ⬜ Upcoming | - |
| Week 6 | FastAPI + Docker | ⬜ Upcoming | - |
| Week 7 | RAGAS + Monitoring | ⬜ Upcoming | - |
| Week 8 | Cloud Deploy | ⬜ Upcoming | - |
| Week 9 | Portfolio Polish | ⬜ Upcoming | - |
| Week 10 | Job Applications | ⬜ Upcoming | - |

---

## Pipeline Flow
PDF File (text or scanned)
│
▼
Smart Extractor
├── PyMuPDF (text PDFs)
└── PaddleOCR (scanned PDFs)
│
▼
GPT-4o Vision
└── Extracts charts, tables, diagrams
│
▼
Text Chunker
└── 512 tokens, 50-token overlap
│
▼
Sentence Transformer Embedder
└── all-MiniLM-L6-v2 → 384-dim vectors
│
▼
Weaviate Vector Database
├── BM25 index (keyword search)
└── Vector index (semantic search)
│
▼
Hybrid RRF Search
└── Combines BM25 + Vector rankings
│
▼
Answer / Edited File

---

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/kashyapbhanderi/doc-intelligence-platform.git
cd doc-intelligence-platform

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment variables
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# 5. Start Weaviate database
docker-compose up -d weaviate

# 6. Run tests
pytest tests/ -v

# 7. Process your PDFs
python scripts/download_direct.py
python scripts/test_batch.py

# 8. Ingest into vector database
python embeddings/ingest.py

# 9. Test search
python scripts/test_hybrid.py
```

---

## Project Structure
doc-intelligence-platform/
├── ingestion/                 # Document processing
│   ├── pdf_extractor.py       # PyMuPDF text extraction
│   ├── ocr_extractor.py       # PaddleOCR for scanned PDFs
│   ├── chunker.py             # 512-token text chunker
│   ├── vision_extractor.py    # GPT-4o Vision integration
│   └── merger.py              # Combines text + vision
├── embeddings/                # Vector operations
│   ├── embedder.py            # Embedding model + Weaviate
│   └── ingest.py              # Batch ingestion pipeline
├── agents/                    # Coming Week 4
├── api/                       # Coming Week 6
├── eval/                      # Evaluation scripts
│   └── search_comparison.py   # Search method comparison
├── scripts/                   # Utility scripts
│   ├── download_direct.py     # Download sample PDFs
│   ├── test_batch.py          # Batch processing
│   ├── test_hybrid.py         # Search comparison
│   └── check_weaviate.py      # Database health check
├── tests/                     # Test suite
│   ├── test_extractor.py      # Chunker tests
│   ├── test_pipeline.py       # Pipeline tests
│   ├── test_vision.py         # Vision tests
│   ├── test_embedder.py       # Embedder tests
│   └── test_search.py         # Search tests
├── data/
│   ├── raw/                   # Downloaded PDFs
│   └── processed/             # Chunked JSON files
├── logs/                      # Pipeline logs
├── docker-compose.yml         # Weaviate + services
├── requirements.txt           # Python dependencies
└── .env                       # API keys (not committed)
