# Build stage: Install dependencies
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Runtime stage: Minimal image with only runtime dependencies
FROM python:3.12-slim

# Install only runtime system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and group
RUN groupadd -r sagemaker -g 1000 && \
    useradd -r -u 1000 -g sagemaker -m -s /bin/bash sagemaker

# Copy virtual environment from builder and set ownership
COPY --from=builder --chown=sagemaker:sagemaker /opt/venv /opt/venv

# Create working directory with proper ownership
RUN mkdir -p /opt/ml/code && \
    chown -R sagemaker:sagemaker /opt/ml

# Set environment variables for SageMaker
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/ml/code:$PYTHONPATH

# Switch to non-root user
USER sagemaker

# Set working directory
WORKDIR /opt/ml/code
