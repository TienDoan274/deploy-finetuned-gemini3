FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libatlas-base-dev \
    gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Install llama-cpp-python with CPU optimizations
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=0" FORCE_CMAKE=1 pip install --force-reinstall llama-cpp-python

# Clone transformers repository at specific version
RUN git clone https://github.com/huggingface/transformers.git \
    && cd transformers \
    && git checkout v4.49.0-Gemma-3 \
    && pip install --no-deps -e .

# Create directory for model
RUN mkdir -p /app/gguf_model

# Copy application files
COPY . /app/

# Expose the port that Gradio runs on
EXPOSE 7860