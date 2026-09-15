FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render (and most platforms) inject a PORT env var at runtime; default to
# 7860 for local/Hugging Face use if PORT isn't set.
ENV PORT=7860
EXPOSE $PORT
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port $PORT"]