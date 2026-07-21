# Makefile — vllatent-ego-drone. Default lane is PURE (numpy/pyyaml; no torch).
# The torch/sports tier (vllatent.encode, vllatent.data, vllatent.model, vllatent.train)
# is exercised in the torch-enabled environment via `make test-torch`.
PY ?= python
SHELL := /bin/bash

# PURE tier (numpy/pyyaml only; CI-importable). The renderer-neutral simulator
# package is pure; the Blender bridge remains outside this gate.
PURE_TIER := vllatent/schemas.py vllatent/config.py vllatent/manifest.py vllatent/ingest/quality.py vllatent/ingest/ego_motion.py vllatent/sim

.PHONY: help setup setup-torch lint typecheck typecheck-all import-smoke test test-torch test-ingest-pure test-ingest-tool encode-smoke blob ralph

help:
	@echo "vllatent-ego-drone dev targets:"
	@echo "  make setup        - light dev deps (ruff, mypy, types-PyYAML, pyyaml, numpy, pytest; NO torch)"
	@echo "  make setup-torch  - the torch/ingest extra (dev box / H20)"
	@echo "  make lint         - ruff check ."
	@echo "  make typecheck    - HARD mypy gate on the PURE tier"
	@echo "  make typecheck-all- mypy on the full package (informational; heavy deps may be missing)"
	@echo "  make import-smoke - import the PURE tier (numpy/pyyaml only, no torch/Blender runtime)"
	@echo "  make test         - pure unit tests (-m 'not torch')"
	@echo "  make test-torch   - torch-tier tests (needs the torch extra)"
	@echo "  make encode-smoke - real-weight DINOv3 forward (downloads ~330MB non-gated timm weights; no token)"
	@echo "  make blob         - pre-commit blob guard"
	@echo "  make ralph        - print the active B3 Ralph-loop launch prompt"

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
	$(PY) -c "import vllatent.schemas, vllatent.config, vllatent.manifest, vllatent.ingest.quality, vllatent.ingest.ego_motion, vllatent.sim.contracts, vllatent.sim.frames, vllatent.sim.skier, vllatent.sim.skier_audit, vllatent.sim.skier_fixtures, vllatent.sim.rig, vllatent.sim.pose, vllatent.sim.labels, vllatent.sim.scene; print('pure import-smoke OK')"

test:
	$(PY) -m pytest -q -m "not torch"

test-torch:
	$(PY) -m pytest -q -m torch

test-ingest-pure:
	$(PY) -m pytest -q tests/test_ingest_*.py -m "not torch and not tool"

test-ingest-tool:
	$(PY) -m pytest -q tests/test_ingest_*.py -m tool

# Downloads timm's NON-GATED DINOv3 ViT-B/16 re-host (~330MB, no token) + runs a real forward.
# From CN, HF_ENDPOINT=https://hf-mirror.com speeds the download: HF_ENDPOINT=... make encode-smoke
encode-smoke:
	$(PY) -m vllatent.encode.dinov3 --smoke

blob:
	bash scripts/check_no_blobs.sh

ralph:
	@bash scripts/ralph.sh
