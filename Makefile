# Makefile — vllatent-ego-drone. Default lane is PURE (numpy/pyyaml; no torch, no sim).
# The torch tier (vllatent.encode, vllatent.data) and sim tier (vllatent.render,
# vllatent.cache) are import-guarded and exercised via `make test-torch` / inside fly0-m1.
PY ?= python
SHELL := /bin/bash

# PURE tier (numpy/pyyaml only; CI-importable). The HARD mypy + import gate scopes here.
PURE_TIER := vllatent/schemas.py vllatent/actions.py vllatent/frames.py vllatent/config.py vllatent/manifest.py vllatent/audit.py

.PHONY: help setup setup-torch lint typecheck typecheck-all import-smoke test test-torch encode-smoke audit blob ralph

help:
	@echo "vllatent-ego-drone dev targets:"
	@echo "  make setup        - light dev deps (ruff, mypy, types-PyYAML, pyyaml, numpy, pytest; NO torch)"
	@echo "  make setup-torch  - the torch extra (dev box / H20: torch, transformers, timm, einops, opencv)"
	@echo "  make lint         - ruff check ."
	@echo "  make typecheck    - HARD mypy gate on the PURE tier"
	@echo "  make typecheck-all- mypy on the full package (informational; torch/sim = missing imports)"
	@echo "  make import-smoke - import the PURE tier (numpy/pyyaml only, no torch/sim)"
	@echo "  make test         - pure unit tests (-m 'not torch and not sim')"
	@echo "  make test-torch   - torch-tier tests (needs the torch extra)"
	@echo "  make encode-smoke - real-weight DINOv3 forward (downloads ~330MB non-gated timm weights; no token)"
	@echo "  make audit        - run the AerialVLN audit parser on the fixture episode (after step 5)"
	@echo "  make blob         - pre-commit blob guard"
	@echo "  make ralph        - print the /ralph-loop launch command"

setup:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install ruff mypy types-PyYAML pyyaml numpy pytest

setup-torch:
	$(PY) -m pip install -e ".[torch]"

lint:
	$(PY) -m ruff check .

typecheck:
	$(PY) -m mypy $(PURE_TIER)

typecheck-all:
	$(PY) -m mypy vllatent || true

import-smoke:
	$(PY) -c "import vllatent.schemas, vllatent.actions, vllatent.frames, vllatent.config, vllatent.manifest, vllatent.audit; print('pure import-smoke OK')"

test:
	$(PY) -m pytest -q -m "not torch and not sim"

test-torch:
	$(PY) -m pytest -q -m torch

# Downloads timm's NON-GATED DINOv3 ViT-B/16 re-host (~330MB, no token) + runs a real forward.
# From CN, HF_ENDPOINT=https://hf-mirror.com speeds the download: HF_ENDPOINT=... make encode-smoke
encode-smoke:
	$(PY) -m vllatent.encode.dinov3 --smoke

audit:
	$(PY) -m vllatent.audit --episode fixtures/episodes/tiny_episode.json

blob:
	bash scripts/check_no_blobs.sh

ralph:
	@bash scripts/ralph.sh
