"""Export the authoritative CPython/NumPy B3-CS3 pose table for Blender.

The isolated Blender bridge consumes this exact float64 table rather than
recomputing SVD/inverse operations under Blender's independently bundled NumPy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vllatent.sim.pose import (
    canonical_pose_export_json_bytes,
    canonical_pose_export_payload,
)
from vllatent.sim.rig import load_rig_manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def export_pose_table(manifest_path: Path, output_path: Path) -> dict[str, object]:
    """Write the exact authoritative table and return its deterministic summary."""
    manifest = load_rig_manifest(manifest_path)
    payload = canonical_pose_export_payload(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_pose_export_json_bytes(manifest))
    return {
        "operation": "export-b3-cs3-pose-table",
        "output_sha256": _sha256_file(output_path),
        "canonical_pose_table_sha256": payload["canonical_pose_table_sha256"],
        "fixture_count": payload["fixture_count"],
        "sample_count": payload["sample_count"],
        "carve_cycle_count": payload["carve_cycle_count"],
        "carve_cycle_sample_count": payload["carve_cycle_sample_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = export_pose_table(args.manifest.resolve(), args.output.resolve())
    print("B3_CS3_EXPORT_POSE_TABLE_OK", json.dumps(summary, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
