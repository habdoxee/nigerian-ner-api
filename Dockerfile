FROM python:3.10-slim

WORKDIR /app

# CPU-only torch build -- the default pip install pulls in the full CUDA
# toolkit (unnecessary here and much larger), which wastes both image size
# and runtime memory on a CPU-only free-tier instance.
RUN pip install --no-cache-dir "torch==2.14.0+cpu" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render (and most platforms) inject a PORT env var at runtime; default to
# 7860 for local/Hugging Face use if PORT isn't set.
ENV PORT=7860
EXPOSE $PORT
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port $PORT"]
