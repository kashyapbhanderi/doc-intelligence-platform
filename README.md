# doc-intelligence-platform

> Production-grade Multimodal Agentic RAG System with Document Editing Agent

![CI](https://github.com/kashyapbhanderi/doc-intelligence-platform/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## What This Project Does

An end-to-end AI system that:
- Ingests PDF, image, and Word documents
- Understands text AND charts/tables (multimodal)
- Answers questions using a 3-agent RAG pipeline
- Edits .docx, .pdf, and image files on command
- Deployed on AWS with live monitoring

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Ingestion | PyMuPDF, PaddleOCR, GPT-4o Vision |
| Embeddings | sentence-transformers, LoRA fine-tuning |
| Vector DB | Weaviate, Hybrid Search (BM25 + dense) |
| Agents | LangGraph, Planner + Executor + Critic |
| Editor | python-docx, PyMuPDF, Pillow |
| API | FastAPI, Docker, streaming SSE |
| MLOps | MLflow, RAGAS, Prometheus, Grafana |

## Project Progress

| Week | Topic | Status |
|------|-------|--------|
| Week 1 | Data Pipeline + OCR + Vision | ✅ Complete |
| Week 2 | Embeddings + Vector DB | ⬜ Starting Monday |
| Week 3 | Fine-tuning | ⬜ Upcoming |
| Week 4 | Multi-Agent System | ⬜ Upcoming |
| Week 5 | Document Editor Agent | ⬜ Upcoming |
| Week 6 | FastAPI + Docker | ⬜ Upcoming |
| Week 7 | RAGAS + Monitoring | ⬜ Upcoming |
| Week 8 | Cloud Deploy | ⬜ Upcoming |

## Quick Start

```bash
# Clone repo
git clone https://github.com/YOUR-USERNAME/doc-intelligence-platform.git
cd doc-intelligence-platform

# Setup environment
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

## Results So Far

| Metric | Value |
|--------|-------|
| Tests passing | 11/11 |
| PDFs supported | Text + Scanned |
| Chunk size | 512 tokens |

## Results So Far

### Week 1 Complete ✅

| Metric | Value |
|--------|-------|
| PDFs downloaded | 52 |
| PDFs processed | YOUR_NUMBER |
| Total chunks created | YOUR_NUMBER |
| Avg chunks per doc | YOUR_NUMBER |
| Tests passing | 17/17 |
| CI pipeline | ✅ Green |
| OCR fallback | ✅ Working |
| Vision LLM | ✅ GPT-4o |

### Pipeline Flow
```
PDF → Smart Extractor → Chunker → Vision → Merger → JSON
         ↓                                    ↓
    (PyMuPDF or OCR)              (text + visual chunks)
```
