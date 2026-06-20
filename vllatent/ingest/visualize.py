"""Per-clip Plotly HTML quality report (B1.9b).

Generates self-contained offline HTML with quality timeline, ego-motion trajectory,
body-frame deltas, VO confidence, latent coherence, and summary table.

All heavy imports (plotly) are LAZY — inside functions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Data extraction helpers (pure numpy, no plotly)
# ---------------------------------------------------------------------------


def compute_latent_coherence(latents: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between consecutive latent frames.

    ``latents``: ``(N, P, D)`` fp16. Returns ``(N-1,)`` f32 cosine similarities.
    """
    n = latents.shape[0]
    coherence = np.zeros(n - 1, dtype=np.float32)
    for i in range(n - 1):
        a = latents[i].astype(np.float32).ravel()
        b = latents[i + 1].astype(np.float32).ravel()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a > 0 and norm_b > 0:
            coherence[i] = float(np.dot(a, b) / (norm_a * norm_b))
        else:
            coherence[i] = 0.0
    return coherence


def compute_cumulative_trajectory(deltas: np.ndarray) -> np.ndarray:
    """Compute 3D positions from body-frame deltas via cumulative sum.

    ``deltas``: ``(N-1, 4)`` [dx, dy, dz, dyaw]. Returns ``(N, 3)`` xyz positions.
    """
    xyz_deltas = deltas[:, :3]
    positions = np.zeros((len(xyz_deltas) + 1, 3), dtype=np.float32)
    positions[1:] = np.cumsum(xyz_deltas, axis=0)
    return positions


def compute_speed_magnitudes(deltas: np.ndarray) -> np.ndarray:
    """Compute per-step speed magnitude from deltas.

    ``deltas``: ``(N-1, 4)`` [dx, dy, dz, dyaw]. Returns ``(N-1,)`` f32.
    """
    xyz = deltas[:, :3]
    return np.linalg.norm(xyz, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Plotly figure builders
# ---------------------------------------------------------------------------


def _fig_quality_timeline(
    frame_quality: np.ndarray,
    timestamps: np.ndarray,
) -> Any:
    """Quality heatmap timeline (RdYlGn colorscale)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=frame_quality,
        mode="lines+markers",
        name="Frame Quality",
        marker={"color": frame_quality, "colorscale": "RdYlGn", "size": 6, "showscale": True,
                "colorbar": {"title": "Quality"}},
        line={"color": "gray", "width": 1},
    ))
    fig.update_layout(
        title="Frame Quality Timeline",
        xaxis_title="Time (s)",
        yaxis_title="Quality Score",
        yaxis_range=[0, 1.05],
        height=300,
    )
    return fig


def _fig_trajectory_3d(
    positions: np.ndarray,
    speeds: np.ndarray,
) -> Any:
    """3D ego-motion trajectory colored by speed."""
    import plotly.graph_objects as go

    fig = go.Figure()
    colors = np.concatenate([[0.0], speeds])
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="lines+markers",
        marker={"size": 3, "color": colors, "colorscale": "Viridis", "showscale": True,
                "colorbar": {"title": "Speed"}},
        line={"color": "lightgray", "width": 2},
        name="3D Trajectory",
    ))
    fig.update_layout(
        title="3D Ego-Motion Trajectory",
        scene={"xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z"},
        height=500,
    )
    return fig


def _fig_deltas(
    deltas: np.ndarray,
    timestamps: np.ndarray,
    frame_quality: np.ndarray,
) -> Any:
    """Body-frame deltas: dx/dy/dz + dyaw + quality overlay."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    t = timestamps[1:]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["Body Deltas (dx, dy, dz)", "Yaw Delta", "Quality Overlay"])

    labels_xyz = ["dx", "dy", "dz"]
    colors_xyz = ["red", "green", "blue"]
    for j, (label, color) in enumerate(zip(labels_xyz, colors_xyz, strict=True)):
        fig.add_trace(go.Scatter(x=t, y=deltas[:, j], mode="lines", name=label,
                                 line={"color": color}), row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=deltas[:, 3], mode="lines", name="dyaw",
                             line={"color": "purple"}), row=2, col=1)

    fig.add_trace(go.Scatter(x=timestamps, y=frame_quality, mode="lines", name="quality",
                             line={"color": "orange"}), row=3, col=1)

    fig.update_layout(height=600, title_text="Body-Frame Deltas")
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    return fig


def _fig_vo_confidence(
    vo_confidence: np.ndarray,
    timestamps: np.ndarray,
    threshold: float = 0.3,
) -> Any:
    """VO confidence timeline with low-confidence threshold."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=vo_confidence,
        mode="lines+markers",
        name="VO Confidence",
        marker={"size": 4},
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Threshold ({threshold})")
    fig.update_layout(
        title="VO Confidence Timeline",
        xaxis_title="Time (s)",
        yaxis_title="Confidence",
        yaxis_range=[0, 1.05],
        height=300,
    )
    return fig


def _fig_latent_coherence(
    coherence: np.ndarray,
    timestamps: np.ndarray,
    threshold: float = 0.85,
) -> Any:
    """Latent coherence (cosine similarity) timeline."""
    import plotly.graph_objects as go

    t = timestamps[1:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t,
        y=coherence,
        mode="lines+markers",
        name="Cosine Similarity",
        marker={"size": 4},
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Scene change ({threshold})")
    fig.update_layout(
        title="Latent Coherence (cos_sim z_t, z_{t+1})",
        xaxis_title="Time (s)",
        yaxis_title="Cosine Similarity",
        yaxis_range=[-0.1, 1.1],
        height=300,
    )
    return fig


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------


def _summary_html(
    clip_id: str,
    n_frames: int,
    duration: float,
    npz_size_kb: float,
    mean_quality: float,
    mean_vo_conf: float,
    mean_coherence: float,
) -> str:
    """HTML summary table."""
    pass_fail = "PASS" if mean_quality >= 0.4 and mean_vo_conf >= 0.3 else "FAIL"
    color = "#2ecc71" if pass_fail == "PASS" else "#e74c3c"
    return f"""
    <div style="margin:20px; padding:15px; border:2px solid {color}; border-radius:8px;">
      <h2>Summary — {clip_id} <span style="color:{color}; font-size:24px;">[{pass_fail}]</span></h2>
      <table style="font-size:16px; border-collapse:collapse;">
        <tr><td style="padding:4px 12px;"><b>Frames</b></td><td>{n_frames}</td></tr>
        <tr><td style="padding:4px 12px;"><b>Duration</b></td><td>{duration:.1f}s</td></tr>
        <tr><td style="padding:4px 12px;"><b>NPZ size</b></td><td>{npz_size_kb:.1f} KB</td></tr>
        <tr><td style="padding:4px 12px;"><b>Mean quality</b></td><td>{mean_quality:.3f}</td></tr>
        <tr><td style="padding:4px 12px;"><b>Mean VO confidence</b></td><td>{mean_vo_conf:.3f}</td></tr>
        <tr><td style="padding:4px 12px;"><b>Mean latent coherence</b></td><td>{mean_coherence:.3f}</td></tr>
      </table>
    </div>
    """


def generate_clip_report(
    npz_path: str | Path,
    *,
    clip_id: str = "",
    out_path: str | Path | None = None,
    vo_threshold: float = 0.3,
    coherence_threshold: float = 0.85,
) -> str:
    """Generate a self-contained HTML quality report for a cached clip.

    Returns the HTML string. If ``out_path`` is given, also writes to that file.
    """
    import plotly.io as pio

    npz_path = Path(npz_path)
    if not clip_id:
        clip_id = npz_path.stem

    with np.load(str(npz_path)) as data:
        latents = data["latents"]
        deltas = data["deltas"]
        vo_confidence = data["vo_confidence"]
        frame_quality = data["frame_quality"]
        timestamps = data["timestamps"]

    n_frames = latents.shape[0]
    duration = float(timestamps[-1] - timestamps[0]) if n_frames > 1 else 0.0
    npz_size_kb = npz_path.stat().st_size / 1024.0

    coherence = compute_latent_coherence(latents)
    positions = compute_cumulative_trajectory(deltas)
    speeds = compute_speed_magnitudes(deltas)

    fig_quality = _fig_quality_timeline(frame_quality, timestamps)
    fig_traj = _fig_trajectory_3d(positions, speeds)
    fig_deltas = _fig_deltas(deltas, timestamps, frame_quality)
    fig_vo = _fig_vo_confidence(vo_confidence, timestamps, threshold=vo_threshold)
    fig_coherence = _fig_latent_coherence(coherence, timestamps, threshold=coherence_threshold)

    summary = _summary_html(
        clip_id=clip_id,
        n_frames=n_frames,
        duration=duration,
        npz_size_kb=npz_size_kb,
        mean_quality=float(np.mean(frame_quality)),
        mean_vo_conf=float(np.mean(vo_confidence)),
        mean_coherence=float(np.mean(coherence)) if len(coherence) > 0 else 0.0,
    )

    sections = [
        pio.to_html(fig_quality, full_html=False, include_plotlyjs="cdn"),
        pio.to_html(fig_traj, full_html=False, include_plotlyjs=False),
        pio.to_html(fig_deltas, full_html=False, include_plotlyjs=False),
        pio.to_html(fig_vo, full_html=False, include_plotlyjs=False),
        pio.to_html(fig_coherence, full_html=False, include_plotlyjs=False),
    ]

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Clip Report — {clip_id}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background: #fafafa; }}
    h1 {{ color: #2c3e50; }}
    .section {{ margin: 20px 0; padding: 10px; background: white; border-radius: 8px;
               box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  </style>
</head>
<body>
  <h1>Quality Report: {clip_id}</h1>
  {summary}
  <div class="section"><h2>Frame Quality Timeline</h2>{sections[0]}</div>
  <div class="section"><h2>3D Ego-Motion Trajectory</h2>{sections[1]}</div>
  <div class="section"><h2>Body-Frame Deltas</h2>{sections[2]}</div>
  <div class="section"><h2>VO Confidence</h2>{sections[3]}</div>
  <div class="section"><h2>Latent Coherence</h2>{sections[4]}</div>
</body>
</html>"""

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html)

    return html


__all__ = [
    "compute_latent_coherence",
    "compute_cumulative_trajectory",
    "compute_speed_magnitudes",
    "generate_clip_report",
]
