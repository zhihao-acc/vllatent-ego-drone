"""Generate architecture diagram as a PNG using matplotlib (no external deps)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path(__file__).resolve().parent / "arch_diagram.png"

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")
fig.patch.set_facecolor("#0d1117")

BOX_KW = dict(boxstyle="round,pad=0.4", linewidth=1.5)
FROZEN = dict(**BOX_KW, facecolor="#1a3a5c", edgecolor="#58a6ff")
TRAIN  = dict(**BOX_KW, facecolor="#3b2e1a", edgecolor="#d29922")
HEAD   = dict(**BOX_KW, facecolor="#1a3a2a", edgecolor="#3fb950")
TEACH  = dict(**BOX_KW, facecolor="#2d1a3a", edgecolor="#bc8cff")
TXT    = dict(color="white", fontsize=9, ha="center", va="center", fontweight="bold")
STXT   = dict(color="#8b949e", fontsize=7, ha="center", va="center")

def box(x, y, label, sub, style, w=2.2, h=1.0):
    ax.text(x, y, label, bbox=style, **TXT)
    if sub:
        ax.text(x, y - 0.45, sub, **STXT)

def arrow(x1, y1, x2, y2, label="", color="#58a6ff"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.2, label, color=color, fontsize=7, ha="center")

# Row 1: Input
box(1.5, 5.8, "RGB 480²", "AirSim render @ GT pose", FROZEN)
box(5.0, 5.8, "DINOv3 ViT-B/16", "FROZEN + CACHED", FROZEN)
box(9.0, 5.8, "z_t  (196×768)", "fp16 latents on disk", FROZEN)

arrow(2.8, 5.8, 3.7, 5.8, "crop→224²")
arrow(6.4, 5.8, 7.6, 5.8, "encode")

# Language branch
box(5.0, 4.3, "CLIP ViT-B/32", "FROZEN text tower", FROZEN)
ax.text(3.0, 4.3, "instruction", color="#8b949e", fontsize=8, ha="center")
arrow(3.7, 4.3, 3.8, 4.3, color="#58a6ff")
arrow(6.3, 4.3, 7.6, 3.3, "cross-attn", color="#58a6ff")

# Row 2: Predictor
box(9.0, 3.3, "Latent Predictor", "~120M, block-causal\nD=768, d=12, h=12", TRAIN, w=2.6)
ax.text(7.2, 2.6, "action (FiLM)", color="#d29922", fontsize=7, ha="center")
arrow(7.6, 2.6, 7.8, 3.0, color="#d29922")
arrow(9.0, 5.2, 9.0, 3.9, "H=3 history", color="#58a6ff")

# Row 3: Heads
box(6.5, 1.3, "4-DoF Waypoint", "Δx,Δy,Δz,Δψ\nNED-body", HEAD)
box(9.0, 1.3, "Trust Head", "p_commit, k*, σ\nsingle-pass", HEAD)
box(11.8, 1.3, "K=5 Ensemble\n+ V-JEPA-2", "offline oracle\n(Phase C)", TEACH)

arrow(8.2, 2.8, 6.5, 1.9, color="#3fb950")
arrow(9.0, 2.8, 9.0, 1.9, color="#3fb950")
arrow(9.8, 2.8, 11.8, 1.9, color="#bc8cff")

# Teacher (distillation)
box(12.5, 4.3, "TrackVLA", "frozen teacher\n(Phase C)", TEACH)
arrow(12.5, 3.8, 11.8, 1.9, "distill", color="#bc8cff")

# Title
ax.text(7, 6.7, "vllatent-ego-drone — Trust-Aware Latent Predictor for Sports-Following Drone",
        color="white", fontsize=12, ha="center", fontweight="bold")

# Legend
legend_items = [
    mpatches.Patch(facecolor="#1a3a5c", edgecolor="#58a6ff", label="Frozen / Cached"),
    mpatches.Patch(facecolor="#3b2e1a", edgecolor="#d29922", label="Trainable (Phase B)"),
    mpatches.Patch(facecolor="#1a3a2a", edgecolor="#3fb950", label="Output Heads"),
    mpatches.Patch(facecolor="#2d1a3a", edgecolor="#bc8cff", label="Teacher / Oracle"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=8,
          facecolor="#161b22", edgecolor="#30363d", labelcolor="white",
          framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {OUT}")
