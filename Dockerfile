FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    PIP_NO_CACHE_DIR=off PYTHONDONTWRITEBYTECODE=1

# 1) OS + SageMath
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates wget gnupg software-properties-common build-essential && \
    add-apt-repository -y universe && \
    apt-get update && \
    apt-get install -y --no-install-recommends sagemath && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Update pip
RUN sage -pip install --upgrade pip

# ─────────────────────────────────────────────
#   Key points
#   • --ignore-installed      ← Ignore existing 1.9
#   • --break-system-packages ← Allow overwriting system packages
# ─────────────────────────────────────────────
RUN sage -pip install \
        --ignore-installed \
        --break-system-packages \
        "sympy>=1.13.3"

# 3) Dependencies
RUN sage -pip install \
        --break-system-packages \
        "torch==2.6.0" \
        "transformers>=4.49.0" \
        "omegaconf>=2.3.0" \
        "wandb>=0.15.11" \
        "accelerate>=0.29.0" \
        "joblib>=1.5.0"

# ▼ If you want the GPU version, replace the torch line above with the following
# RUN sage -pip install --break-system-packages \
#     --extra-index-url https://download.pytorch.org/whl/cu124 \
#     "torch==2.6.0"

# 4) Editable install of transformer_algebra
COPY . /app/
RUN sage -pip install -e /app

