"""Plot 3D reference trajectories from AerialVLN episodes."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "aerialvln_json" / "train.slice.json"
OUT = Path(__file__).resolve().parent / "trajectory_3d.png"

with open(DATA) as f:
    raw = json.load(f)
episodes = raw["episodes"] if isinstance(raw, dict) else raw

sorted_eps = sorted(episodes, key=lambda e: len(e["reference_path"]), reverse=True)
picks = [sorted_eps[0], sorted_eps[len(sorted_eps)//4], sorted_eps[len(sorted_eps)//2]]

fig = plt.figure(figsize=(14, 7))
fig.patch.set_facecolor("#0d1117")

COLORS = ["#58a6ff", "#3fb950", "#d29922"]

# Left: 3D view
ax3d = fig.add_subplot(121, projection="3d")
ax3d.set_facecolor("#0d1117")
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
ax3d.xaxis.pane.set_edgecolor("#30363d")
ax3d.yaxis.pane.set_edgecolor("#30363d")
ax3d.zaxis.pane.set_edgecolor("#30363d")
ax3d.tick_params(colors="#8b949e")
ax3d.xaxis.label.set_color("#8b949e")
ax3d.yaxis.label.set_color("#8b949e")
ax3d.zaxis.label.set_color("#8b949e")

for ep, color in zip(picks, COLORS):
    path = np.array(ep["reference_path"])
    x, y, z = path[:, 0], path[:, 1], path[:, 2]
    eid = ep["episode_id"]
    n = len(path)
    ax3d.plot(x, y, z, color=color, lw=1.5, alpha=0.8, label=f"ep {eid} ({n} steps)")
    ax3d.scatter(x[0], y[0], z[0], color=color, s=60, marker="o", zorder=5)
    ax3d.scatter(x[-1], y[-1], z[-1], color=color, s=60, marker="x", zorder=5)

ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Z")
ax3d.set_title("3D Reference Trajectories", color="white", fontsize=11)
ax3d.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=8)

# Right: top-down with instructions
ax2d = fig.add_subplot(122)
ax2d.set_facecolor("#161b22")
ax2d.tick_params(colors="#8b949e")
for spine in ax2d.spines.values():
    spine.set_edgecolor("#30363d")

y_text = 0.95
for ep, color in zip(picks, COLORS):
    path = np.array(ep["reference_path"])
    x, y = path[:, 0], path[:, 1]
    eid = ep["episode_id"]
    ax2d.plot(x, y, color=color, lw=1.8, alpha=0.8)
    ax2d.scatter(x[0], y[0], color=color, s=60, marker="o", zorder=5)
    ax2d.scatter(x[-1], y[-1], color=color, s=60, marker="x", zorder=5)

    instr = ep.get("instruction", {})
    text = instr.get("instruction_text", str(instr))[:80]
    ax2d.text(0.02, y_text, f"ep {eid}: \"{text}...\"",
              transform=ax2d.transAxes, color=color, fontsize=7,
              va="top", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.2", facecolor="#0d1117", alpha=0.7))
    y_text -= 0.08

ax2d.set_xlabel("X", color="#8b949e")
ax2d.set_ylabel("Y", color="#8b949e")
ax2d.set_title("Top-Down View + Instructions", color="white", fontsize=11)

fig.suptitle("AerialVLN — Ground-Truth Reference Paths",
             color="white", fontsize=13, fontweight="bold", y=1.0)

fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {OUT}")
