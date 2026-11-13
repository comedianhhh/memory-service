FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/models

WORKDIR /app

# Dependencies first: this layer is reused until requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake both ONNX models into the image so a container start never downloads ~210 MB.
RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; TextEmbedding('BAAI/bge-small-en-v1.5'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

# Code last: changing it only rebuilds from here.
COPY alembic.ini .
COPY migrations/ migrations/
COPY memory_service/ memory_service/

# Run as a non-root user.
RUN useradd --create-home app
USER app

EXPOSE 8000

# Migrate (idempotent), then replace the shell with uvicorn so it receives SIGTERM directly.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn memory_service.api:app --host 0.0.0.0 --port 8000"]
