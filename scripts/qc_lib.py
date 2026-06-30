"""Shared QC-report helpers (data-quality visualization).

Pure-ish viz layer used by both ``validate_megasam.py`` (trajectory only) and
``qc_report.py`` (combined frame-filter + trajectory report + index). Imports
numpy / plotly / PIL only — no torch. Lives in ``scripts/`` so the package
pure-tier stays untouched.

Two outcomes this renders:
  1. Content-filter accept/reject per frame (FPV vs non-FPV), as a decision
     filmstrip + thumbnail contact sheet.
  2. MegaSaM trajectory + VO verdict, as an interactive Plotly figure.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np

# Decision colors (content filter)
COLOR_FPV = "#2e7d32"        # green  — kept (FPV)
COLOR_OBJECT = "#c62828"     # red    — rejected: YOLO object (drone/camera/overlay)
COLOR_STATIC = "#ef6c00"     # orange — rejected: too little motion
COLOR_SHORT = "#f9a825"      # amber  — rejected: FPV-ish but in a too-short run
COLOR_OTHER = "#757575"      # gray   — rejected: other / non-FPV shot

REASON_LABELS = {
    "fpv": "FPV (kept)",
    "object": "object",
    "static": "static",
    "short": "short-run",
    "other": "non-FPV",
}
REASON_COLORS = {
    "fpv": COLOR_FPV,
    "object": COLOR_OBJECT,
    "static": COLOR_STATIC,
    "short": COLOR_SHORT,
    "other": COLOR_OTHER,
}

_MOTION_THRESHOLD = 8.0  # mirror content_filter._MOTION_THRESHOLD


# ---------------------------------------------------------------------------
# Per-frame decision classification (content filter only)
# ---------------------------------------------------------------------------


def classify_frame_reasons(
    fpv_mask: np.ndarray,
    motion_scores: np.ndarray | None,
    rejected_objects: np.ndarray | None,
    *,
    motion_threshold: float = _MOTION_THRESHOLD,
) -> list[str]:
    """Label every frame with WHY the content filter kept/dropped it.

    Categories: ``fpv`` (kept), ``object`` (YOLO rejected), ``static`` (below
    motion threshold), ``short`` (had motion + no object but lived in a run too
    short to survive), ``other`` (dropped for any remaining reason).
    """
    n = len(fpv_mask)
    reasons: list[str] = []
    for i in range(n):
        if fpv_mask[i]:
            reasons.append("fpv")
            continue
        if rejected_objects is not None and bool(rejected_objects[i]):
            reasons.append("object")
        elif motion_scores is not None and float(motion_scores[i]) < motion_threshold:
            reasons.append("static")
        elif motion_scores is not None and rejected_objects is not None:
            # had motion and no object yet still dropped => short-run pruning
            reasons.append("short")
        else:
            reasons.append("other")
    return reasons


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _data_uri_png(arr: np.ndarray) -> str:
    """Encode an (H,W,3) uint8 array as a base64 PNG data URI."""
    from PIL import Image

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _thumb_data_uri(path: Path, width: int = 144) -> str:
    """Load an image, downscale to ``width`` px, return a base64 PNG data URI."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if img.width > width:
        h = int(img.height * width / img.width)
        img = img.resize((width, h))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def filmstrip_data_uri(reasons: list[str], *, height: int = 28) -> str:
    """Render a 1-cell-per-frame decision strip as a base64 PNG data URI."""
    n = max(1, len(reasons))
    strip = np.zeros((height, n, 3), dtype=np.uint8)
    for i, r in enumerate(reasons):
        hexcol = REASON_COLORS.get(r, COLOR_OTHER).lstrip("#")
        rgb = tuple(int(hexcol[j : j + 2], 16) for j in (0, 2, 4))
        strip[:, i, :] = rgb
    return _data_uri_png(strip)


# ---------------------------------------------------------------------------
# Trajectory / VO figure (Plotly) — Plotly-6 safe
# ---------------------------------------------------------------------------


def _safe_hline(fig, y: float, row: int, col: int, x: np.ndarray, color: str, label: str) -> None:
    """Draw a horizontal threshold line WITHOUT add_hline.

    ``add_hline(row,col)`` scans every trace for ``.xaxis`` to test subplot
    emptiness, which raises on a Scatter3d (3D scene) under Plotly 6. Drawing
    the line as a normal 2D trace via add_trace avoids that scan entirely.
    """
    import plotly.graph_objects as go

    if len(x) == 0:
        return
    x0, x1 = float(np.min(x)), float(np.max(x))
    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[y, y],
            mode="lines",
            line=dict(color=color, dash="dash", width=1),
            name=label,
            hovertext=label,
            hoverinfo="text",
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def build_vo_figure(
    *,
    poses: np.ndarray,
    confidences: np.ndarray,
    deltas: np.ndarray,
    decision: str,
    clip_id: str,
    fps: float,
    max_speed_ms: float,
    max_yaw_rate_deg_s: float,
    low_confidence: float,
):
    """Build the interactive MegaSaM VO figure (3D trajectory + signal panels)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    positions = poses[:, :3, 3]
    n = len(positions)
    speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) * fps if n > 1 else np.array([])
    yaw_rates = np.abs(deltas[:, 3]) * fps if len(deltas) > 0 else np.array([])

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "scene", "colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=[
            "3D Trajectory (colored by speed)",
            f"Speed (m/s) — limit {max_speed_ms:g}",
            f"Yaw Rate (°/s) — limit {max_yaw_rate_deg_s:g}",
            f"VO Confidence — low < {low_confidence:g}",
            "Acceleration Magnitude (m/s²)",
        ],
        vertical_spacing=0.08,
    )

    speed_colors = np.zeros(n)
    speed_colors[1:] = speeds
    fig.add_trace(
        go.Scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            mode="lines+markers",
            marker=dict(size=2, color=speed_colors, colorscale="RdYlGn_r",
                        colorbar=dict(title="Speed (m/s)", x=1.02)),
            line=dict(width=2, color="gray"),
            name="Trajectory",
        ),
        row=1, col=1,
    )

    frames = np.arange(len(speeds))
    fig.add_trace(go.Scatter(x=frames, y=speeds, mode="lines", name="Speed",
                             line=dict(color="steelblue")), row=2, col=1)
    _safe_hline(fig, max_speed_ms, 2, 1, frames, "red", f"max {max_speed_ms:g} m/s")

    if len(yaw_rates) > 0:
        xr = np.arange(len(yaw_rates))
        fig.add_trace(go.Scatter(x=xr, y=yaw_rates, mode="lines", name="Yaw Rate",
                                 line=dict(color="orange")), row=2, col=2)
        _safe_hline(fig, max_yaw_rate_deg_s, 2, 2, xr, "red", f"max {max_yaw_rate_deg_s:g}°/s")

    xc = np.arange(len(confidences))
    fig.add_trace(go.Scatter(x=xc, y=confidences, mode="lines", name="Confidence",
                             line=dict(color="green")), row=3, col=1)
    _safe_hline(fig, low_confidence, 3, 1, xc, "red", f"low {low_confidence:g}")

    if n > 2:
        velocity = np.diff(positions, axis=0) * fps
        acceleration = np.diff(velocity, axis=0) * fps
        accel_mag = np.linalg.norm(acceleration, axis=1)
        fig.add_trace(go.Scatter(x=np.arange(len(accel_mag)), y=accel_mag, mode="lines",
                                 name="Accel Mag", line=dict(color="purple")), row=3, col=2)

    verdict_color = {"GO": "green", "CONDITIONAL-GO": "orange", "NO-GO": "red"}.get(decision, "gray")
    fig.update_layout(
        title=f"MegaSaM VO — {clip_id} — <span style='color:{verdict_color}'>{decision}</span>",
        height=900,
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# HTML page templating
# ---------------------------------------------------------------------------

_PAGE_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px;
       background: #fafafa; color: #222; }
h1 { font-size: 20px; } h2 { font-size: 16px; margin-top: 28px; }
.badge { display:inline-block; padding:2px 8px; border-radius:4px; color:#fff;
         font-size:12px; font-weight:600; }
.legend span { margin-right: 14px; font-size: 12px; }
.legend i { display:inline-block; width:12px; height:12px; border-radius:2px;
            margin-right:4px; vertical-align:middle; }
.filmstrip { width:100%; image-rendering: pixelated; height:28px;
             border:1px solid #ccc; }
.grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(150px,1fr));
        gap:8px; margin-top:10px; }
.cell { border:3px solid #ccc; border-radius:4px; overflow:hidden; background:#fff; }
.cell img { width:100%; display:block; }
.cap { font-size:11px; padding:2px 4px; color:#444; }
table { border-collapse: collapse; width:100%; font-size:13px; }
th,td { border:1px solid #ddd; padding:6px 8px; text-align:left; }
th { background:#eee; } tr:nth-child(even){ background:#f3f3f3; }
a { color:#1565c0; text-decoration:none; } a:hover{ text-decoration:underline; }
.small { color:#777; font-size:12px; }
"""


def html_page(title: str, body: str) -> str:
    """Wrap a body fragment in a minimal self-contained HTML document."""
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{_PAGE_CSS}</style></head>"
        f"<body>{body}</body></html>"
    )


def legend_html(reasons_present: list[str]) -> str:
    """Render a small color legend for the decision categories present."""
    parts = ["<div class='legend'>"]
    for r in ["fpv", "object", "static", "short", "other"]:
        if r in reasons_present:
            parts.append(
                f"<span><i style='background:{REASON_COLORS[r]}'></i>{REASON_LABELS[r]}</span>"
            )
    parts.append("</div>")
    return "".join(parts)


def contact_sheet_html(
    frame_paths: list[Path],
    reasons: list[str],
    motion_scores: np.ndarray | None,
    *,
    max_thumbs: int = 120,
    thumb_width: int = 144,
) -> tuple[str, int]:
    """Build the thumbnail contact sheet. Returns (html, stride_used)."""
    n = len(frame_paths)
    stride = max(1, (n + max_thumbs - 1) // max_thumbs)
    idxs = list(range(0, n, stride))
    cells = []
    for i in idxs:
        reason = reasons[i] if i < len(reasons) else "other"
        color = REASON_COLORS.get(reason, COLOR_OTHER)
        uri = _thumb_data_uri(frame_paths[i], width=thumb_width)
        motion = f"{float(motion_scores[i]):.1f}" if motion_scores is not None and i < len(motion_scores) else "?"
        cap = f"#{i} · {REASON_LABELS.get(reason, reason)} · m={motion}"
        cells.append(
            f"<div class='cell' style='border-color:{color}'>"
            f"<img src='{uri}'><div class='cap'>{cap}</div></div>"
        )
    return f"<div class='grid'>{''.join(cells)}</div>", stride


__all__ = [
    "classify_frame_reasons",
    "filmstrip_data_uri",
    "build_vo_figure",
    "html_page",
    "legend_html",
    "contact_sheet_html",
    "REASON_COLORS",
    "REASON_LABELS",
]
