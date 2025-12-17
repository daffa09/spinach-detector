FROM python:3.11-slim

WORKDIR /app

# Runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY main.py .
COPY models/*.onnx models/

EXPOSE 5005

CMD ["python", "main.py"]
