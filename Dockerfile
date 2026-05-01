FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    ca-certificates \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && python3 -m pip install uv

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen

COPY . .

RUN chmod +x /app/entrypoint.sh
CMD ["/app/entrypoint.sh"]
