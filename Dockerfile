FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_DATA_DIR=/app/data \
    HF_HOME=/app/data/huggingface \
    TORCH_HOME=/app/data/torch \
    TRANSFORMERS_CACHE=/app/data/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY .streamlit ./.streamlit
COPY streamlit_app.py ./
COPY README.md ./
RUN mkdir -p /app/data/huggingface /app/data/torch

EXPOSE 8503
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8503"]
