FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy clean requirements
COPY requirements.docker.txt .

# Install packages — use clean requirements only
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.docker.txt

# Copy app code
COPY . .

# Create directories
RUN mkdir -p data/uploads data/processed \
    data/raw logs models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s \
    --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health \
    || exit 1

CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]