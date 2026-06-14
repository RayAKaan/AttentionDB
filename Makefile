.PHONY: build test bench clean fmt clippy

build:
	cargo build --workspace

test:
	cargo test --workspace

bench:
	cargo bench --workspace

clean:
	cargo clean

fmt:
	cargo fmt --all

clippy:
	cargo clippy --workspace -- -D warnings

storage:
	cd storage && cargo run --release

hnsw:
	cd hnsw && cargo run --release

query:
	cd query && cargo run --release

multihead:
	cd multihead && cargo run --release

api:
	cd api && cargo run --release --bin attentiondb-server

learned:
	cd learned && cargo run --release

distributed:
	cd distributed && cargo run --release
