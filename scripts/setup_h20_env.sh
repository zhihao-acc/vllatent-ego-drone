#!/usr/bin/env bash
# PRINT the AutoDL H20 conda+pip setup block — the USER pastes it on the H20.
# SSH is HANDS-OFF: the agent never drives ssh; it produces the block, the user runs it.
#
# Builds the training conda env on top of the saved navdreamer/Wan2.1 mirror
# (torch 2.8 / CUDA 12.x already present). See vault [[environment-and-equipment]].
set -euo pipefail

cat <<'EOF'
# === Run on the AutoDL H20 instance (NVLink ~96 GB), reusing the saved navdreamer mirror ===
export HF_ENDPOINT=https://hf-mirror.com

conda create -y -n vllatent-ego-drone python=3.10
conda activate vllatent-ego-drone

# torch 2.8 + CUDA 12.x: prefer the mirror's existing build; else install matching wheels.
pip install "transformers>=4.56" "timm>=1.0.20" einops opencv-python numpy pyyaml

# Install vllatent (cached-latent training stack) from the synced repo:
#   rsync the repo up first (HANDS-OFF: user runs rsync), then:
# pip install -e ".[torch]"

python -c "import torch, transformers, timm; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
EOF
