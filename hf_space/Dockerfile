# Hugging Face Docker Space: one process, one port. ct-bot serves the JSON API
# and the static Next.js build together on 7860.
# Build context is the bundle prepare.sh assembles (rust/, frontend/, the model).

# --- frontend: static export ---------------------------------------------------
FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# Empty API URL = same origin: the page calls /api/* on the server that served it.
ENV NEXT_OUTPUT=export NEXT_PUBLIC_API_URL=""
RUN npx next build

# --- engine --------------------------------------------------------------------
FROM rust:1.92-slim-trixie AS engine
# ort's build script downloads the prebuilt ONNX Runtime over TLS; g++ provides
# the libstdc++ the statically linked runtime needs at link time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config ca-certificates libssl-dev g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY rust/ ./rust/
RUN cargo build --release --manifest-path rust/Cargo.toml -p ct-bot

# --- runtime -------------------------------------------------------------------
FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* && useradd -m -u 1000 user
WORKDIR /app
COPY --from=engine /build/rust/target/release/ct-bot /usr/local/bin/ct-bot
COPY --from=web /web/out ./web
COPY model.int8.onnx ./model.int8.onnx
USER user

# MCTS_SIMS can be overridden from the Space settings (Variables) without a
# rebuild. On the free 2-vCPU tier 800 sims measured 3-6 s a move.
ENV PORT=7860 MCTS_SIMS=800 ORT_THREADS=2 STATIC_DIR=/app/web RUST_LOG=info
EXPOSE 7860
CMD ["sh", "-c", "exec ct-bot serve --model /app/model.int8.onnx --threads \"$ORT_THREADS\""]
