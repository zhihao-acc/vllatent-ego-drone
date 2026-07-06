# Makefile — vllatent-ego-drone. Default lane is PURE (numpy/pyyaml; no torch).
# The torch/sports tier (vllatent.encode, vllatent.data, vllatent.model, vllatent.train)
# is import-guarded and exercised via `make test-torch`.
PY ?= python
SHELL := /bin/bash

# PURE tier (numpy/pyyaml only; CI-importable). The HARD mypy + import gate scopes here.
PURE_TIER := vllatent/schemas.py vllatent/actions.py vllatent/frames.py vllatent/config.py vllatent/manifest.py vllatent/audit.py vllatent/ingest/quality.py vllatent/ingest/ego_motion.py

.PHONY: help setup setup-torch lint typecheck typecheck-all import-smoke test test-torch test-ingest-pure test-ingest-tool encode-smoke vjepa-smoke text-smoke audit blob ralph

help:
	@echo "vllatent-ego-drone dev targets:"
	@echo "  make setup        - light dev deps (ruff, mypy, types-PyYAML, pyyaml, numpy, pytest; NO torch)"
	@echo "  make setup-torch  - the torch extra (dev box / H20: torch, transformers, timm, einops, opencv)"
	@echo "  make lint         - ruff check ."
	@echo "  make typecheck    - HARD mypy gate on the PURE tier"
	@echo "  make typecheck-all- mypy on the full package (informational; heavy deps may be missing)"
	@echo "  make import-smoke - import the PURE tier (numpy/pyyaml only, no torch/sim)"
	@echo "  make test         - pure unit tests (-m 'not torch and not sim')"
	@echo "  make test-torch   - torch-tier tests (needs the torch extra)"
	@echo "  make encode-smoke - real-weight DINOv3 forward (downloads ~330MB non-gated timm weights; no token)"
	@echo "  make vjepa-smoke  - real-weight V-JEPA-2 surprise (downloads ~1.30GB non-gated ViT-L; no token)"
	@echo "  make text-smoke   - real-weight CLIP text tower -> (M,768) lang_tokens (downloads CLIP; no token)"
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
	$(PY) -m pytest -q -m "not torch and not sim" --ignore=tests/test_data_shapes.py

test-torch:
	$(PY) -m pytest -q -m torch --ignore=tests/test_data_shapes.py

test-ingest-pure:
	$(PY) -m pytest -q tests/test_ingest_*.py -m "not torch and not sim and not tool"

test-ingest-tool:
	$(PY) -m pytest -q tests/test_ingest_*.py -m tool

# Downloads timm's NON-GATED DINOv3 ViT-B/16 re-host (~330MB, no token) + runs a real forward.
# From CN, HF_ENDPOINT=https://hf-mirror.com speeds the download: HF_ENDPOINT=... make encode-smoke
encode-smoke:
	$(PY) -m vllatent.encode.dinov3 --smoke

# Downloads Meta's NON-GATED V-JEPA-2 ViT-L (~1.30GB safetensors, no token) + runs a real surprise.
# From CN, HF_ENDPOINT=https://hf-mirror.com speeds the download: HF_ENDPOINT=... make vjepa-smoke
vjepa-smoke:
	$(PY) -m vllatent.verify.vjepa2 --smoke

# Downloads the NON-GATED CLIP ViT-B/32 text tower (no token) + runs a real text encode -> (M,768).
# From CN, HF_ENDPOINT=https://hf-mirror.com speeds the download: HF_ENDPOINT=... make text-smoke
text-smoke:
	$(PY) -m vllatent.encode.text --smoke

audit:
	$(PY) -m vllatent.audit --episode fixtures/episodes/tiny_episode.json

blob:
	bash scripts/check_no_blobs.sh

ralph:
	@bash scripts/ralph.sh
