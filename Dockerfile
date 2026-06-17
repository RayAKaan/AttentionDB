# ============================================================================
# AttentionDB — Multi-Stage Production Dockerfile
# ============================================================================
# Stage 1: Build (with protoc)
# Stage 2: Runtime (minimal image)
# ============================================================================

FROM rust:1.96-bookworm AS builder

RUN apt-get update && apt-get install -y protobuf-compiler && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cargo build --workspace --release

FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y ca-certificates curl && rm -rf /var/lib/apt/lists/*

RUN groupadd -r attentiondb && useradd -r -g attentiondb -m attentiondb

COPY --from=builder /app/target/release/attentiondb-server /usr/local/bin/
COPY --from=builder /app/target/release/attentiondb-repl /usr/local/bin/

RUN mkdir -p /data && chown attentiondb:attentiondb /data

USER attentiondb
WORKDIR /data

EXPOSE 7400 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["attentiondb-server"]
