FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -e ".[openai]"

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

# Wait for postgres then start API (compose healthcheck also gates start)
CMD ["uvicorn", "mem01.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
