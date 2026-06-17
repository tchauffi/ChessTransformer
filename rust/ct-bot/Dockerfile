# Rust backend for the ChessTransformer web app (replaces backend/Dockerfile).
# Build from the repo root:  docker build -f rust/ct-bot/Dockerfile .

FROM rust:1.92-slim-trixie AS builder

# ort's build script downloads the prebuilt ONNX Runtime over TLS (via ureq +
# native-tls, hence libssl-dev/pkg-config at build time).
# g++ provides libstdc++, which the statically-linked ONNX Runtime needs at link time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config ca-certificates libssl-dev g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY rust/ ./rust/
RUN cargo build --release --manifest-path rust/Cargo.toml -p ct-bot

FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/rust/target/release/ct-bot /usr/local/bin/ct-bot
# ONNX Runtime (CPU) is statically linked; only the model is needed at runtime.
COPY data/models/pos2move_v2.1/model_ema.onnx ./model_ema.onnx

EXPOSE 5001
ENV RUST_LOG=info

CMD ["ct-bot", "serve", "--model", "/app/model_ema.onnx", "--port", "5001"]
