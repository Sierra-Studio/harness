# Harness — CLI setup & dev tasks.
# Run `make help` for the list. Requires: uv (https://docs.astral.sh/uv/) and Docker.

# NOTE: docker-compose.yml exposes Postgres on host port 5433 (mapped 5433:5432).
# Your .env DATABASE_URL host port MUST be 5433, e.g.:
#   DATABASE_URL=postgresql://harness:harness@localhost:5433/harness
COMPOSE ?= docker compose
UV      ?= uv

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# One-shot setup
# ---------------------------------------------------------------------------

.PHONY: setup
setup: install env db-up db-init ## Full setup: deps + .env + Postgres + schema
	@echo "✅ Harness ready. Run 'make chat' to start an interactive session."

.PHONY: install
install: ## Create .venv and install deps (incl. dev extras)
	$(UV) sync --extra dev

.PHONY: env
env: ## Create .env from .env.example if missing
	@test -f .env || (cp .env.example .env && echo "Created .env — set OPENROUTER_API_KEY and confirm DATABASE_URL uses port 5433")

# ---------------------------------------------------------------------------
# Database (Postgres + pgvector)
# ---------------------------------------------------------------------------

.PHONY: db-up
db-up: ## Start Postgres + pgvector (waits for healthy)
	$(COMPOSE) up -d
	@echo "Waiting for Postgres to become healthy..."
	@until [ "$$($(COMPOSE) ps -q db | xargs docker inspect -f '{{.State.Health.Status}}' 2>/dev/null)" = "healthy" ]; do sleep 1; done
	@echo "Postgres is healthy."

.PHONY: db-init
db-init: ## Apply schema.sql to the database
	$(UV) run harness init-db

.PHONY: db-down
db-down: ## Stop Postgres (keeps data volume)
	$(COMPOSE) down

.PHONY: db-reset
db-reset: ## Wipe Postgres data volume, restart, re-apply schema
	$(COMPOSE) down -v
	$(MAKE) db-up db-init

.PHONY: db-logs
db-logs: ## Tail Postgres logs
	$(COMPOSE) logs -f db

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

.PHONY: chat
chat: ## Start an interactive harness session
	$(UV) run harness chat

.PHONY: serve
serve: ## Start the SSE HTTP server
	$(UV) run harness serve

.PHONY: demo
demo: ## Offline demo — no DB, no API key (in-memory repo + fake provider)
	DATABASE_URL="" OPENROUTER_API_KEY="" $(UV) run harness chat

# ---------------------------------------------------------------------------
# Dev
# ---------------------------------------------------------------------------

.PHONY: test
test: ## Run the test suite
	$(UV) run pytest -q

.PHONY: lint
lint: ## Lint with ruff
	$(UV) run ruff check harness tests

.PHONY: fmt
fmt: ## Format with ruff
	$(UV) run ruff format harness tests

.PHONY: typecheck
typecheck: ## Type-check with mypy
	$(UV) run mypy

.PHONY: check
check: lint typecheck test ## Lint + typecheck + test

.PHONY: docs
docs: ## Serve docs locally (http://127.0.0.1:8000)
	$(UV) sync --extra docs
	$(UV) run mkdocs serve
