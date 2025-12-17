# ==== MULTI-STAGE BUILD FOR SLIM IMAGE ====
# Stage 1: Builder - Install dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install PyTorch CPU-only (much smaller than GPU version)
RUN pip install --no-cache-dir --user \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime - Minimal final image
FROM python:3.11-slim

WORKDIR /app

# Install only runtime libraries (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy only necessary application files
COPY main.py .
COPY models/*.pt models/

# Set Python path
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose Flask port
EXPOSE 5005

# Run the Flask app
CMD ["python", "main.py"]
