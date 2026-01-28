# Voice Agent ASR - Lightweight Dockerfile
# Uses CUDA runtime image (~4GB vs ~20GB for full PyTorch image)

FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set Python environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.12-venv \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA 12.4 support (compatible with CUDA 13.0)
# RUN pip3 install --no-cache-dir \
#    torch==2.6.0 \
#    torchaudio==2.6.0 \
#    --index-url https://download.pytorch.org/whl/cu124

RUN python3 -m venv env

# Install other dependencies
COPY requirements.txt .
RUN env/bin/pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY voice_agent_rnnt.py .

# Create model cache directory
RUN mkdir -p /app/models

# Environment variables
ENV MODEL_REPO="milind-plivo/parakeet-multilingual-base" \
    MODEL_FILENAME="parakeet-rnnt-1.1b-multilingual.nemo" \
    MODEL_CACHE_DIR="/app/models" \
    HF_HOME="/app/models/hf_cache"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["env/bin/python", "voice_agent_rnnt.py"]