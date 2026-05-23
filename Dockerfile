# Multi-stage build — smaller final image
FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker cache layer)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
     pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/uploads data/processed \
    data/raw logs models

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
    --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health \
    || exit 1

# Start command
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]