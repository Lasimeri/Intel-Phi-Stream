# Top-level entry points. Run `make help` for the list.

.DEFAULT_GOAL := help
# The llama.cpp checkout whose headers are bound and whose CUDA build is
# linked (build.md); phi-stream is skipped with a note without the headers.
LLAMA_CPP_DIR ?= $(HOME)/llama.cpp
export LLAMA_CPP_DIR
HAVE := $(wildcard $(LLAMA_CPP_DIR)/include/llama.h)
NOTE = $(if $(HAVE),,@echo "phi-stream skipped: no $(LLAMA_CPP_DIR)/include/llama.h (set LLAMA_CPP_DIR)")

.PHONY: help build test fmt clippy docs-check check clean

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

build: ## Build phi-stream (release) where llama.cpp is found
	$(if $(HAVE),cargo build --release,$(NOTE))

test: ## Run the tests that need no model
	$(if $(HAVE),cargo test --release,$(NOTE))

fmt: ## Check formatting
	cargo fmt --all -- --check

clippy: ## Lint
	$(if $(HAVE),cargo clippy --release --all-targets -- -D warnings,$(NOTE))

docs-check: ## Enforce sibling .md files, the no-dash rule and relative links
	scripts/check-docs.sh

check: docs-check fmt clippy build test ## Everything CI would run

clean: ## Remove build outputs
	cargo clean
