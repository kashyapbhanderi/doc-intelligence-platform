"""
api/models.py
Pydantic models for request/response validation.

Pydantic automatically:
- Validates incoming request data
- Returns clear errors for bad input
- Generates OpenAPI documentation
Industry standard for FastAPI apps.
"""
from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Question to ask the RAG system"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve"
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list
    is_faithful: bool
    sub_queries: list
    latency_seconds: float


class EditRequest(BaseModel):
    instruction: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Natural language edit instruction"
    )
    file_path: str = Field(
        ...,
        description="Path to file to edit"
    )


class EditResponse(BaseModel):
    instruction: str
    result: str
    success: bool


class IngestResponse(BaseModel):
    filename: str
    status: str
    chunks: int
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    weaviate: str
    total_chunks: int