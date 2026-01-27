# Voice Agent ASR - Dockerfile
# Multi-stage build for smaller final image

FROM nvcr.io/nvidia/pytorch:24.01-py3 AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY voice_agent_rnnt.py .

# Create model cache directory
RUN mkdir -p /app/models

# Environment variables for configuration (can be overridden)
ENV MODEL_REPO="milind-plivo/parakeet-multilingual-base" \
    MODEL_FILENAME="parakeet-rnnt-1.1b-multilingual.nemo" \
    MODEL_CACHE_DIR="/app/models" \
    HF_HOME="/app/models/hf_cache"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["python", "voice_agent_rnnt.py"]
