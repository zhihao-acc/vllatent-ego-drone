#!/usr/bin/env bash
# Run the full 3-step MegaSaM pipeline on a single clip.
#
# MegaSaM is a 3-step pipeline:
#   Step 1: Depth-Anything (mono disparity)
#   Step 2: UniDepth (metric depth + FoV)
#   Step 3: Camera tracking (SLAM)
#
# Usage:
#   bash scripts/run_megasam_pipeline.sh \
#     --clip-id ski01 \
#     --frames-dir ingest_data/frames/ski01 \
#     --megasam-dir ~/CODE/MegaSaM \
#     [--gpu 0] [--encoder vitl] [--out-dir ingest_data/frames/ski01_megasam]
#
# Output: copies reconstructions/{clip_id}/ to {out_dir}/ for our ingest pipeline.
set -euo pipefail

CLIP_ID=""
FRAMES_DIR=""
MEGASAM_DIR=""
GPU="0"
ENCODER="vitl"
OUT_DIR=""
CONDA_ENV="mega_sam"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clip-id)     CLIP_ID="$2"; shift 2 ;;
        --frames-dir)  FRAMES_DIR="$2"; shift 2 ;;
        --megasam-dir) MEGASAM_DIR="$2"; shift 2 ;;
        --gpu)         GPU="$2"; shift 2 ;;
        --encoder)     ENCODER="$2"; shift 2 ;;
        --out-dir)     OUT_DIR="$2"; shift 2 ;;
        --conda-env)   CONDA_ENV="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --clip-id ID --frames-dir DIR --megasam-dir DIR [--gpu N] [--out-dir DIR]"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$CLIP_ID" || -z "$FRAMES_DIR" || -z "$MEGASAM_DIR" ]]; then
    echo "[megasam] ERROR: --clip-id, --frames-dir, and --megasam-dir are required"
    exit 1
fi

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="$(dirname "$FRAMES_DIR")/${CLIP_ID}_megasam"
fi

FRAMES_DIR="$(realpath "$FRAMES_DIR")"
MEGASAM_DIR="$(realpath "$MEGASAM_DIR")"
# Resolve OUT_DIR to absolute BEFORE the `cd "$MEGASAM_DIR"` below — otherwise the
# relative mkdir/cp at the end land the output INSIDE the MegaSaM repo instead of
# ours, and parse_megasam_output (which uses the caller-relative path) finds nothing.
OUT_DIR="$(realpath -m "$OUT_DIR")"

echo "[megasam] =============================="
echo "[megasam] Pipeline: ${CLIP_ID}"
echo "[megasam] Frames:   ${FRAMES_DIR}"
echo "[megasam] MegaSaM:  ${MEGASAM_DIR}"
echo "[megasam] GPU:      ${GPU}"
echo "[megasam] Output:   ${OUT_DIR}"
echo "[megasam] =============================="

N_FRAMES=$(find "$FRAMES_DIR" -maxdepth 1 -name "*.jpg" -o -name "*.png" | wc -l)
echo "[megasam] Found ${N_FRAMES} frames"
if [[ "$N_FRAMES" -lt 2 ]]; then
    echo "[megasam] ERROR: need at least 2 frames"
    exit 1
fi

cd "$MEGASAM_DIR"

DA_CKPT="Depth-Anything/checkpoints/depth_anything_${ENCODER}14.pth"
if [[ ! -f "$DA_CKPT" ]]; then
    echo "[megasam] ERROR: DepthAnything checkpoint not found: ${DA_CKPT}"
    exit 1
fi

MEGASAM_CKPT="checkpoints/megasam_final.pth"
if [[ ! -f "$MEGASAM_CKPT" ]]; then
    echo "[megasam] ERROR: MegaSaM checkpoint not found: ${MEGASAM_CKPT}"
    exit 1
fi

# Step 1: Depth-Anything (mono disparity)
echo ""
echo "[megasam] === Step 1/3: Depth-Anything ==="
CUDA_VISIBLE_DEVICES="$GPU" XFORMERS_DISABLED=1 conda run -n "$CONDA_ENV" \
    python Depth-Anything/run_videos.py \
        --encoder "$ENCODER" \
        --load-from "$DA_CKPT" \
        --img-path "$FRAMES_DIR" \
        --outdir "Depth-Anything/video_visualization/${CLIP_ID}"
echo "[megasam] Step 1 done."

# Step 2: UniDepth (metric depth + FoV)
echo ""
echo "[megasam] === Step 2/3: UniDepth ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIDEPTH_WRAPPER="${SCRIPT_DIR}/megasam_shims/run_unidepth.py"
export PYTHONPATH="${PYTHONPATH:-}:${MEGASAM_DIR}/UniDepth"
CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$CONDA_ENV" \
    python "$UNIDEPTH_WRAPPER" \
        --scene-name "$CLIP_ID" \
        --img-path "$FRAMES_DIR" \
        --outdir UniDepth/outputs
echo "[megasam] Step 2 done."

# Step 3: Camera tracking
echo ""
echo "[megasam] === Step 3/3: Camera tracking ==="
CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$CONDA_ENV" \
    python camera_tracking_scripts/test_demo.py \
        --datapath "$FRAMES_DIR" \
        --weights "$MEGASAM_CKPT" \
        --scene_name "$CLIP_ID" \
        --mono_depth_path Depth-Anything/video_visualization \
        --metric_depth_path UniDepth/outputs \
        --disable_vis
echo "[megasam] Step 3 done."

# Copy output to our pipeline's expected location
RECON_DIR="${MEGASAM_DIR}/reconstructions/${CLIP_ID}"
if [[ ! -d "$RECON_DIR" ]]; then
    echo "[megasam] ERROR: reconstruction output not found at ${RECON_DIR}"
    exit 1
fi

mkdir -p "$OUT_DIR"
cp -v "${RECON_DIR}/poses.npy" "$OUT_DIR/"
cp -v "${RECON_DIR}/motion_prob.npy" "$OUT_DIR/" 2>/dev/null || true
cp -v "${RECON_DIR}/intrinsics.npy" "$OUT_DIR/" 2>/dev/null || true

DROID_NPZ="${MEGASAM_DIR}/outputs/${CLIP_ID}_droid.npz"
if [[ -f "$DROID_NPZ" ]]; then
    cp -v "$DROID_NPZ" "$OUT_DIR/"
fi

echo ""
echo "[megasam] =============================="
echo "[megasam] Pipeline complete: ${CLIP_ID}"
echo "[megasam] Output: ${OUT_DIR}"
echo "[megasam] Files:"
ls -lh "$OUT_DIR"
echo "[megasam] =============================="
