# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.10 CLI project using a `src/` layout. Core code lives in `src/rag_demo/`, with the Typer entrypoint in `cli.py` and feature modules such as `rag.py`, `milvus_bench.py`, `qdrant_weaviate.py`, `chunking_recall.py`, and `embedding_selection.py`. Tests live in `tests/` and mirror feature areas. Long-form materials are in `docs/` and `tutorial/`. Vector database and profiling support is split between `docker-compose.yml` and `observability/`.

## Build, Test, and Development Commands

- `uv sync`: install dependencies from `pyproject.toml` and `uv.lock`.
- `uv run rag-demo doctor`: validate the local CLI environment.
- `uv run rag-demo t1 index --docs ./confluence-export --persist ./.rag/index`: index Markdown content for the RAG demo.
- `uv run rag-demo t1 ask "question" --persist ./.rag/index`: run a one-shot query.
- `docker-compose up -d`: start Milvus, Qdrant, Weaviate, and related services for vector DB demos.
- `uv run pytest`: run the test suite with the configured quiet pytest output.
- `uv run ruff check .`: run linting and import-order checks.

## Coding Style & Naming Conventions

Use Ruff as the source of truth: 100-character line length, Python 3.10 target, and rules `E`, `F`, `I`, `UP`, `B`, and `SIM`. Prefer typed functions and small feature-focused modules. Use `snake_case` for modules, functions, variables, and CLI helpers. Keep CLI commands consistent with the existing `t1`, `t2`, `t3`, etc. task grouping.

## Testing Guidelines

Tests use pytest and are discovered from `tests/` via `pyproject.toml`. Name files `test_<feature>.py` and test functions `test_<behavior>()`. Favor deterministic tests with small fixtures, `tmp_path`, and explicit assertions for validation errors. For vector database workflows, keep automated tests lightweight; use Docker-backed commands for manual or integration checks.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, for example `Add RAG tutorial docs` and `Fix Weaviate class handling`. Follow that style: start with `Add`, `Fix`, `Update`, or a similar verb, and keep the subject focused. Pull requests should include a concise summary, verification commands, required services or environment variables, and links to related issues or docs. Include screenshots or report snippets when changing CLI output, tutorials, or benchmark reporting.

## Security & Configuration Tips

Do not commit secrets, API keys, local Ollama endpoints, generated `.rag/` data, or benchmark artifacts. Prefer documenting configuration in README or docs files. eBPF observability commands may require `sudo`; keep those scripts narrow.

## Code Search

Use `semble search` to find code by describing what it does or naming a symbol/identifier, instead of grep:

​```bash
semble search "authentication flow" ./my-project
semble search "save_pretrained" ./my-project
semble search "save model to disk" ./my-project --top-k 10
​```

Use `semble find-related` to discover code similar to a known location (pass `file_path` and `line` from a prior search result):

​```bash
semble find-related src/auth.py 42 ./my-project
​```

`path` defaults to the current directory when omitted; git URLs are accepted.

If `semble` is not on `$PATH`, use `uvx --from "semble[mcp]" semble` in its place.

## Workflow

1. Start with `semble search` to find relevant chunks.
2. Inspect full files only when the returned chunk is not enough context.
3. Optionally use `semble find-related` with a promising result's `file_path` and `line` to discover related implementations.
4. Use grep only when you need exhaustive literal matches or quick confirmation of an exact string.
