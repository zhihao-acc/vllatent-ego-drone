"""Renderer-neutral contracts for the B3-CS causal ski simulator.

This module is PURE: stdlib and NumPy only.  It owns the canonical simulator
command record, the nine initial camera interventions, the fixed camera
contract, sibling identity, and deterministic serialization/hashing.  The
historical six-field passive-video token is intentionally not imported here.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final, Literal

import numpy as np


def _immutable_buffer(value: np.ndarray, dtype: np.dtype[np.generic]) -> np.ndarray:
    """Copy an array onto an immutable bytes buffer, preserving shape and order."""
    expected = np.dtype(dtype)
    contiguous = np.ascontiguousarray(value, dtype=expected)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=expected).reshape(contiguous.shape)


CONTRACT_SCHEMA_VERSION: Final[str] = "b3-cs1-camera-contract-v1"
SKIER_DIGEST_SCHEMA_VERSION: Final[str] = "b3-cs-skier-digest-v1"
SKIER_ROOT_SCHEMA_VERSION: Final[str] = "b3-cs2-skier-root-v1"
SKIER_POSE_ROOT_SCHEMA_VERSION: Final[str] = "b3-cs3-skier-root-pose-v1"

HORIZON_STEPS: Final[int] = 8
FIXED_DT_SECONDS: Final[float] = 0.2
COMMAND_DIM: Final[int] = 4
COMMAND_FIELDS: Final[tuple[str, ...]] = (
    "v_forward_m_s",
    "v_right_m_s",
    "v_down_m_s",
    "yaw_rate_rad_s",
)

YAW_RATE_MAGNITUDE_RAD_S: Final[float] = math.pi / 15.0
FORWARD_SPEED_MAGNITUDE_M_S: Final[float] = 1.0
LATERAL_SPEED_MAGNITUDE_M_S: Final[float] = 0.75
VERTICAL_SPEED_MAGNITUDE_M_S: Final[float] = 0.50

R_CAM_FROM_RIG: Final[np.ndarray] = _immutable_buffer(
    np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]],
        dtype=np.float64,
    ),
    np.dtype("<f8"),
)
R_RIG_FROM_CAM: Final[np.ndarray] = _immutable_buffer(R_CAM_FROM_RIG.T, np.dtype("<f8"))
T_CAM_FROM_RIG_M: Final[np.ndarray] = _immutable_buffer(np.zeros(3, dtype=np.float64), np.dtype("<f8"))
T_RIG_FROM_CAM_M: Final[np.ndarray] = _immutable_buffer(np.zeros(3, dtype=np.float64), np.dtype("<f8"))

RENDER_WIDTH_PX: Final[int] = 224
RENDER_HEIGHT_PX: Final[int] = 224
LENS_MM: Final[float] = 24.0
SENSOR_WIDTH_MM: Final[float] = 36.0
SENSOR_FIT: Final[str] = "HORIZONTAL"
SHIFT_X: Final[float] = 0.0
SHIFT_Y: Final[float] = 0.0
PIXEL_ASPECT_X: Final[float] = 1.0
PIXEL_ASPECT_Y: Final[float] = 1.0
CLIP_START_M: Final[float] = 0.1
CLIP_END_M: Final[float] = 500.0
DEPTH_OF_FIELD_ENABLED: Final[bool] = False

_CAMERA_K: Final[np.ndarray] = _immutable_buffer(
    np.array(
        [
            [
                LENS_MM / SENSOR_WIDTH_MM * RENDER_WIDTH_PX,
                0.0,
                RENDER_WIDTH_PX / 2.0,
            ],
            [
                0.0,
                LENS_MM / SENSOR_WIDTH_MM * RENDER_WIDTH_PX,
                RENDER_HEIGHT_PX / 2.0,
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    np.dtype("<f8"),
)

_FORBIDDEN_SKIER_KEY_TOKENS: Final[tuple[str, ...]] = (
    "branch",
    "camera",
    "command",
    "image",
    "mask",
    "pixel",
    "record_valid",
    "render",
    "rgb",
    "visibility",
    "visible",
)

_CS2_ROOT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "absolute_tick",
        "position_xy_m",
        "heading_rad",
        "speed_m_s",
        "acceleration_m_s2",
        "curvature_1_m",
        "omega_rad_s",
        "gross_lean_rad",
        "T_world_from_groundroot",
        "T_world_from_armature",
        "tracked_joint_positions_root_m",
    }
)
_CS2_SKIS_KEYS: Final[frozenset[str]] = frozenset(
    {
        "dimensions_m",
        "stance_half_width_m",
        "centerline_ordering_m",
        "inner_tip_gap_m",
        "left",
        "right",
    }
)
_CS2_ONE_SKI_KEYS: Final[frozenset[str]] = frozenset(
    {
        "side",
        "attack_rad",
        "edge_rad",
        "centerline_origin_world_m",
        "base_origin_world_m",
        "binding_origin_world_m",
        "contact_origin_world_m",
        "target_F_world_from_ski",
        "realized_F_world_from_ski",
        "analytic_slip_longitudinal_lateral_m_s",
        "realized_slip_longitudinal_lateral_m_s",
        "realized_attack_rad",
        "realized_edge_rad",
        "frame_orientation_residual_rad",
    }
)
_CS2_CONTACT_KEYS: Final[frozenset[str]] = frozenset({"left_contact_origin_world_m", "right_contact_origin_world_m"})
_CS2_PHASE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "maneuver_id",
        "maneuver_phase",
        "animation_clip_ids",
        "animation_phase",
        "animation_blend_weights",
    }
)
_RANDOMNESS_KEYS: Final[frozenset[str]] = frozenset({"seed"})


class BranchId(str, Enum):
    """The exact ordered nine initial camera interventions."""

    ZERO = "zero"
    YAW_PLUS = "yaw_plus"
    YAW_MINUS = "yaw_minus"
    FORWARD_PLUS = "forward_plus"
    FORWARD_MINUS = "forward_minus"
    LATERAL_PLUS = "lateral_plus"
    LATERAL_MINUS = "lateral_minus"
    VERTICAL_PLUS = "vertical_plus"
    VERTICAL_MINUS = "vertical_minus"


BRANCH_IDS: Final[tuple[str, ...]] = tuple(branch.value for branch in BranchId)

_EXPECTED_COMMAND_ROWS: Final[dict[BranchId, tuple[float, float, float, float]]] = {
    BranchId.ZERO: (0.0, 0.0, 0.0, 0.0),
    BranchId.YAW_PLUS: (0.0, 0.0, 0.0, YAW_RATE_MAGNITUDE_RAD_S),
    BranchId.YAW_MINUS: (0.0, 0.0, 0.0, -YAW_RATE_MAGNITUDE_RAD_S),
    BranchId.FORWARD_PLUS: (FORWARD_SPEED_MAGNITUDE_M_S, 0.0, 0.0, 0.0),
    BranchId.FORWARD_MINUS: (-FORWARD_SPEED_MAGNITUDE_M_S, 0.0, 0.0, 0.0),
    BranchId.LATERAL_PLUS: (0.0, LATERAL_SPEED_MAGNITUDE_M_S, 0.0, 0.0),
    BranchId.LATERAL_MINUS: (0.0, -LATERAL_SPEED_MAGNITUDE_M_S, 0.0, 0.0),
    BranchId.VERTICAL_PLUS: (0.0, 0.0, VERTICAL_SPEED_MAGNITUDE_M_S, 0.0),
    BranchId.VERTICAL_MINUS: (0.0, 0.0, -VERTICAL_SPEED_MAGNITUDE_M_S, 0.0),
}


class DatasetSplit(str, Enum):
    """Dataset split carried by an indivisible root/sibling group."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


ImageField = Literal["cx", "cy", "log_h"]


@dataclass(frozen=True)
class ExpectedImageEffect:
    """Expected scalar image effect relative to the matched zero branch."""

    field: ImageField
    sign: Literal[-1, 1]


_IMAGE_EFFECTS: Final[dict[BranchId, ExpectedImageEffect]] = {
    BranchId.YAW_PLUS: ExpectedImageEffect("cx", -1),
    BranchId.YAW_MINUS: ExpectedImageEffect("cx", 1),
    BranchId.FORWARD_PLUS: ExpectedImageEffect("log_h", 1),
    BranchId.FORWARD_MINUS: ExpectedImageEffect("log_h", -1),
    BranchId.LATERAL_PLUS: ExpectedImageEffect("cx", -1),
    BranchId.LATERAL_MINUS: ExpectedImageEffect("cx", 1),
    BranchId.VERTICAL_PLUS: ExpectedImageEffect("cy", -1),
    BranchId.VERTICAL_MINUS: ExpectedImageEffect("cy", 1),
}


def _nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name}: expected str, got {type(value).__name__}")
    normalized = unicodedata.normalize("NFC", value)
    if not normalized:
        raise ValueError(f"{name}: expected non-empty string")
    return normalized


def _immutable_array(
    name: str,
    value: object,
    shape: tuple[int, ...],
    dtype: np.dtype[np.generic],
    *,
    finite: bool = True,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name}: expected np.ndarray, got {type(value).__name__}")
    expected = np.dtype(dtype)
    if value.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {value.shape}")
    if value.dtype != expected:
        raise ValueError(f"{name}: expected dtype {expected}, got {value.dtype}")
    if finite and not np.all(np.isfinite(value)):
        raise ValueError(f"{name}: expected all finite values")
    canonical = np.array(value, dtype=expected, order="C", copy=True)
    if canonical.dtype.kind == "f":
        canonical[canonical == 0.0] = 0.0
    return _immutable_buffer(canonical, expected)


def _f64(name: str, value: object, shape: tuple[int, ...]) -> np.ndarray:
    return _immutable_array(name, value, shape, np.dtype("<f8"))


def _bool(name: str, value: object, shape: tuple[int, ...]) -> np.ndarray:
    return _immutable_array(name, value, shape, np.dtype(np.bool_), finite=False)


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name}: expected real scalar, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name}: expected finite value")
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True, eq=False)
class BranchProgram:
    """One eight-step requested camera program in canonical float64 SI units."""

    branch_id: BranchId
    requested_command: np.ndarray
    dt_seconds: np.ndarray
    record_valid: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.branch_id, BranchId):
            raise TypeError(f"branch_id: expected BranchId, got {type(self.branch_id).__name__}")
        object.__setattr__(
            self,
            "requested_command",
            _f64(
                "requested_command",
                self.requested_command,
                (HORIZON_STEPS, COMMAND_DIM),
            ),
        )
        expected_row = np.asarray(_EXPECTED_COMMAND_ROWS[self.branch_id], dtype=np.float64)
        expected_command = np.repeat(expected_row[None, :], HORIZON_STEPS, axis=0)
        if not np.array_equal(self.requested_command, expected_command):
            raise ValueError(
                f"requested_command: must equal the frozen eight-step program for branch {self.branch_id.value!r}"
            )
        dt = _f64("dt_seconds", self.dt_seconds, (HORIZON_STEPS,))
        if not np.array_equal(dt, np.full(HORIZON_STEPS, FIXED_DT_SECONDS)):
            raise ValueError(f"dt_seconds: every row must equal fixed {FIXED_DT_SECONDS} seconds")
        object.__setattr__(self, "dt_seconds", dt)
        valid = _bool("record_valid", self.record_valid, (HORIZON_STEPS,))
        if not bool(np.all(valid)):
            raise ValueError("record_valid: all nine pilot programs are valid at every step")
        object.__setattr__(self, "record_valid", valid)

    def model_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return separate float32 command and dt tensors without normalization."""
        command = self.requested_command.astype(np.float32, copy=True)
        dt = self.dt_seconds.astype(np.float32, copy=True)
        command.setflags(write=False)
        dt.setflags(write=False)
        return command, dt


def _constant_program(branch_id: BranchId, command: Sequence[float]) -> BranchProgram:
    row = np.asarray(command, dtype=np.float64)
    requested = np.repeat(row[None, :], HORIZON_STEPS, axis=0)
    return BranchProgram(
        branch_id=branch_id,
        requested_command=requested,
        dt_seconds=np.full(HORIZON_STEPS, FIXED_DT_SECONDS, dtype=np.float64),
        record_valid=np.ones(HORIZON_STEPS, dtype=np.bool_),
    )


def canonical_branch_programs() -> tuple[BranchProgram, ...]:
    """Return the frozen zero and signed single-axis branch catalog."""
    return tuple(_constant_program(branch_id, _EXPECTED_COMMAND_ROWS[branch_id]) for branch_id in BranchId)


def program_by_id(branch_id: BranchId) -> BranchProgram:
    """Return one canonical program by typed branch ID."""
    if not isinstance(branch_id, BranchId):
        raise TypeError(f"branch_id: expected BranchId, got {type(branch_id).__name__}")
    return canonical_branch_programs()[tuple(BranchId).index(branch_id)]


def expected_image_effect(branch_id: BranchId) -> ExpectedImageEffect:
    """Return the preregistered sign for one nonzero branch."""
    if not isinstance(branch_id, BranchId):
        raise TypeError(f"branch_id: expected BranchId, got {type(branch_id).__name__}")
    if branch_id is BranchId.ZERO:
        raise ValueError("branch_id: zero has no signed image effect")
    return _IMAGE_EFFECTS[branch_id]


def sign_eligible(
    *,
    plus_optical_depth_m: float,
    minus_optical_depth_m: float,
    plus_center_xy: np.ndarray,
    minus_center_xy: np.ndarray,
    plus_displacement_px: float,
    minus_displacement_px: float,
) -> bool:
    """Apply the exact depth/center/one-pixel signed-geometry eligibility rule."""
    plus_depth = _finite_float("plus_optical_depth_m", plus_optical_depth_m)
    minus_depth = _finite_float("minus_optical_depth_m", minus_optical_depth_m)
    plus_center = _f64("plus_center_xy", plus_center_xy, (2,))
    minus_center = _f64("minus_center_xy", minus_center_xy, (2,))
    plus_displacement = _finite_float("plus_displacement_px", plus_displacement_px)
    minus_displacement = _finite_float("minus_displacement_px", minus_displacement_px)
    centers_ok = bool(
        np.all((plus_center >= 0.1) & (plus_center <= 0.9)) and np.all((minus_center >= 0.1) & (minus_center <= 0.9))
    )
    return bool(
        plus_depth > 2.0
        and minus_depth > 2.0
        and centers_ok
        and abs(plus_displacement) > 1.0
        and abs(minus_displacement) > 1.0
    )


@dataclass(frozen=True)
class RootSiblingIdentity:
    """One branch identity in an indivisible root/split group."""

    root_id: str
    split_group_id: str
    branch_id: BranchId
    split: DatasetSplit

    def __post_init__(self) -> None:
        root_id = _nonempty_string("root_id", self.root_id)
        split_group = _nonempty_string("split_group_id", self.split_group_id)
        if split_group != root_id:
            raise ValueError("split_group_id: must equal root_id")
        if not isinstance(self.branch_id, BranchId):
            raise TypeError(f"branch_id: expected BranchId, got {type(self.branch_id).__name__}")
        if not isinstance(self.split, DatasetSplit):
            raise TypeError(f"split: expected DatasetSplit, got {type(self.split).__name__}")
        object.__setattr__(self, "root_id", root_id)
        object.__setattr__(self, "split_group_id", split_group)


def validate_sibling_group(identities: Sequence[RootSiblingIdentity]) -> None:
    """Require exactly one complete nine-branch root in exactly one split."""
    if not isinstance(identities, Sequence):
        raise TypeError(f"identities: expected sequence, got {type(identities).__name__}")
    if len(identities) != len(BranchId):
        raise ValueError(f"identities: expected exactly {len(BranchId)} siblings")
    for index, identity in enumerate(identities):
        if not isinstance(identity, RootSiblingIdentity):
            raise TypeError(f"identities[{index}]: expected RootSiblingIdentity, got {type(identity).__name__}")
    roots = {identity.root_id for identity in identities}
    groups = {identity.split_group_id for identity in identities}
    splits = {identity.split for identity in identities}
    branches = [identity.branch_id for identity in identities]
    if len(roots) != 1 or len(groups) != 1:
        raise ValueError("identities: siblings must share one root_id and split_group_id")
    if len(splits) != 1:
        raise ValueError("identities: sibling group crosses dataset splits")
    if set(branches) != set(BranchId) or len(set(branches)) != len(BranchId):
        raise ValueError("identities: branch catalog must contain each of the nine branches once")


@dataclass(frozen=True, eq=False)
class CameraContract:
    """Fixed camera axes, intrinsics, lens, and crop settings hashed with actions."""

    K: np.ndarray
    R_cam_from_rig: np.ndarray
    R_rig_from_cam: np.ndarray
    t_cam_from_rig_m: np.ndarray
    t_rig_from_cam_m: np.ndarray

    def __post_init__(self) -> None:
        intrinsic = _f64("K", self.K, (3, 3))
        cam_from_rig = _f64("R_cam_from_rig", self.R_cam_from_rig, (3, 3))
        rig_from_cam = _f64("R_rig_from_cam", self.R_rig_from_cam, (3, 3))
        t_cam = _f64("t_cam_from_rig_m", self.t_cam_from_rig_m, (3,))
        t_rig = _f64("t_rig_from_cam_m", self.t_rig_from_cam_m, (3,))
        if not np.array_equal(cam_from_rig, R_CAM_FROM_RIG):
            raise ValueError("R_cam_from_rig: does not match the frozen FRD-to-camera matrix")
        if not np.array_equal(rig_from_cam, R_RIG_FROM_CAM):
            raise ValueError("R_rig_from_cam: does not match R_cam_from_rig.T")
        if not np.array_equal(rig_from_cam, cam_from_rig.T):
            raise ValueError("R_rig_from_cam: must be the exact transpose")
        if not np.array_equal(t_cam, T_CAM_FROM_RIG_M) or not np.array_equal(t_rig, T_RIG_FROM_CAM_M):
            raise ValueError("camera/rig translations: must be exactly zero")
        if not np.array_equal(intrinsic[2], np.array([0.0, 0.0, 1.0])):
            raise ValueError("K: expected homogeneous bottom row [0,0,1]")
        if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
            raise ValueError("K: focal lengths must be positive")
        if not np.array_equal(intrinsic, _CAMERA_K):
            raise ValueError("K: does not match the frozen 24-mm 224-square intrinsics")
        object.__setattr__(self, "K", intrinsic)
        object.__setattr__(self, "R_cam_from_rig", cam_from_rig)
        object.__setattr__(self, "R_rig_from_cam", rig_from_cam)
        object.__setattr__(self, "t_cam_from_rig_m", t_cam)
        object.__setattr__(self, "t_rig_from_cam_m", t_rig)

    def manifest(self) -> dict[str, object]:
        """Return every frozen field entering the camera-contract hash."""
        return {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "K": self.K,
            "R_cam_from_rig": self.R_cam_from_rig,
            "R_rig_from_cam": self.R_rig_from_cam,
            "t_cam_from_rig_m": self.t_cam_from_rig_m,
            "t_rig_from_cam_m": self.t_rig_from_cam_m,
            "lens_mm": LENS_MM,
            "sensor_width_mm": SENSOR_WIDTH_MM,
            "sensor_fit": SENSOR_FIT,
            "render_width_px": RENDER_WIDTH_PX,
            "render_height_px": RENDER_HEIGHT_PX,
            "shift_x": SHIFT_X,
            "shift_y": SHIFT_Y,
            "pixel_aspect_x": PIXEL_ASPECT_X,
            "pixel_aspect_y": PIXEL_ASPECT_Y,
            "clip_start_m": CLIP_START_M,
            "clip_end_m": CLIP_END_M,
            "depth_of_field_enabled": DEPTH_OF_FIELD_ENABLED,
            "crop": "full_224_square_noop",
            "blender_camera_axes": "+X right,+Y up,-Z optical-forward",
        }


def default_camera_contract() -> CameraContract:
    """Construct the frozen 24-mm/36-mm horizontal-fit 224-square camera."""
    return CameraContract(
        K=_CAMERA_K,
        R_cam_from_rig=R_CAM_FROM_RIG,
        R_rig_from_cam=R_RIG_FROM_CAM,
        t_cam_from_rig_m=T_CAM_FROM_RIG_M,
        t_rig_from_cam_m=T_RIG_FROM_CAM_M,
    )


def _canonical_array(value: np.ndarray) -> dict[str, object]:
    if value.dtype.kind == "f":
        if value.dtype.itemsize not in (4, 8):
            raise TypeError(f"canonical array: unsupported floating dtype {value.dtype}")
        dtype = np.dtype(f"<f{value.dtype.itemsize}")
    elif value.dtype.kind in "iu":
        dtype = np.dtype(f"<{value.dtype.kind}{value.dtype.itemsize}")
    elif value.dtype == np.dtype(np.bool_):
        dtype = np.dtype(np.bool_)
    else:
        raise TypeError(f"canonical array: unsupported dtype {value.dtype}")
    # Preserve the v1 scalar-as-one-element canonical shape while taking a
    # writable copy for signed-zero normalization.
    array = np.ascontiguousarray(value.astype(dtype, copy=False)).copy()
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ValueError("canonical array: expected finite values")
    if array.dtype.kind == "f":
        array[array == 0.0] = 0.0
    return {
        "$ndarray": {
            "data_hex": array.tobytes(order="C").hex(),
            "dtype": dtype.str,
            "shape": [int(size) for size in array.shape],
        }
    }


def _canonical_value(value: object) -> object:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return {"$int": str(int(value))}
    if isinstance(value, (float, np.floating)):
        number = _finite_float("canonical float", value)
        return _canonical_array(np.asarray(number, dtype=np.float64))
    if isinstance(value, np.ndarray):
        return _canonical_array(value)
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"canonical key: expected str, got {type(key).__name__}")
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise ValueError(f"canonical key: normalization collision for {key!r}")
            normalized[normalized_key] = _canonical_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    raise TypeError(f"canonical value: unsupported type {type(value).__name__}")


def canonical_bytes(value: Mapping[str, object]) -> bytes:
    """Serialize a mapping with sorted NFC keys and little-endian numeric bytes."""
    if not isinstance(value, Mapping):
        raise TypeError(f"value: expected mapping, got {type(value).__name__}")
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_canonical(value: Mapping[str, object]) -> str:
    """Return SHA-256 over :func:`canonical_bytes`."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def camera_contract_sha256(
    program: BranchProgram,
    camera: CameraContract | None = None,
) -> str:
    """Hash canonical action arrays together with the complete camera contract."""
    if not isinstance(program, BranchProgram):
        raise TypeError(f"program: expected BranchProgram, got {type(program).__name__}")
    contract = default_camera_contract() if camera is None else camera
    if not isinstance(contract, CameraContract):
        raise TypeError(f"camera: expected CameraContract, got {type(contract).__name__}")
    return sha256_canonical(
        {
            **contract.manifest(),
            "branch_id": program.branch_id,
            "requested_command": program.requested_command,
            "dt_seconds": program.dt_seconds,
            "record_valid": program.record_valid,
        }
    )


def _reject_forbidden_skier_keys(value: object, path: str = "skier") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path}: expected string keys")
            normalized = unicodedata.normalize("NFC", key).lower()
            if any(token in normalized for token in _FORBIDDEN_SKIER_KEY_TOKENS):
                raise ValueError(f"{path}.{key}: observation/camera fields cannot enter skier digest")
            _reject_forbidden_skier_keys(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden_skier_keys(item, f"{path}[{index}]")


def _require_exact_keys(
    domain: str,
    value: Mapping[str, object],
    allowed: tuple[frozenset[str], ...],
) -> frozenset[str]:
    keys = frozenset(value)
    if keys not in allowed:
        expected = " or ".join(str(sorted(candidate)) for candidate in allowed)
        raise ValueError(f"skier.{domain}: keys must exactly match a typed digest schema; expected {expected}")
    return keys


def _require_f64_digest_array(path: str, value: object, shape: tuple[int | None, ...]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"skier.{path}: expected np.ndarray")
    if value.dtype != np.dtype("<f8"):
        raise ValueError(f"skier.{path}: expected little-endian float64")
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected for actual, expected in zip(value.shape, shape, strict=True)
    ):
        raise ValueError(f"skier.{path}: expected shape {shape}, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"skier.{path}: expected finite values")
    return value


def _require_digest_float(path: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, np.floating)):
        raise TypeError(f"skier.{path}: expected float scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"skier.{path}: expected finite value")
    return result


def _require_digest_int(path: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"skier.{path}: expected integer")
    return int(value)


def _require_digest_text(path: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"skier.{path}: expected non-empty string")
    return value


def _validate_skier_digest_domains(payload: Mapping[str, object]) -> None:
    mappings: dict[str, Mapping[str, object]] = {}
    for domain in ("root", "skis", "contacts", "phases", "randomness"):
        value = payload[domain]
        if not isinstance(value, Mapping):
            raise TypeError(f"skier.{domain}: expected mapping")
        mappings[domain] = value
    root = mappings["root"]
    skis = mappings["skis"]
    contacts = mappings["contacts"]
    phases = mappings["phases"]
    randomness = mappings["randomness"]
    _require_exact_keys("root", root, (_CS2_ROOT_KEYS,))
    _require_exact_keys("skis", skis, (_CS2_SKIS_KEYS,))
    _require_exact_keys("contacts", contacts, (_CS2_CONTACT_KEYS,))
    _require_exact_keys("phases", phases, (_CS2_PHASE_KEYS,))
    _require_exact_keys("randomness", randomness, (_RANDOMNESS_KEYS,))
    _require_digest_int("randomness.seed", randomness["seed"])
    root_schema_version = _require_digest_text("root.schema_version", root["schema_version"])
    supported_root_versions = {SKIER_ROOT_SCHEMA_VERSION, SKIER_POSE_ROOT_SCHEMA_VERSION}
    if root_schema_version not in supported_root_versions:
        raise ValueError(f"skier.root.schema_version: expected one of {sorted(supported_root_versions)!r}")
    pose_count: int | None = 17 if root_schema_version == SKIER_POSE_ROOT_SCHEMA_VERSION else None
    _require_f64_digest_array(
        "local_bone_transforms",
        payload["local_bone_transforms"],
        (pose_count, 4, 4),
    )
    _require_digest_int("root.absolute_tick", root["absolute_tick"])
    _require_f64_digest_array("root.position_xy_m", root["position_xy_m"], (2,))
    for field in (
        "heading_rad",
        "speed_m_s",
        "acceleration_m_s2",
        "curvature_1_m",
        "omega_rad_s",
        "gross_lean_rad",
    ):
        _require_digest_float(f"root.{field}", root[field])
    for field in ("T_world_from_groundroot", "T_world_from_armature"):
        _require_f64_digest_array(f"root.{field}", root[field], (4, 4))
    _require_f64_digest_array(
        "root.tracked_joint_positions_root_m",
        root["tracked_joint_positions_root_m"],
        (pose_count, 3),
    )
    _require_f64_digest_array("skis.dimensions_m", skis["dimensions_m"], (3,))
    for field in (
        "stance_half_width_m",
        "centerline_ordering_m",
        "inner_tip_gap_m",
    ):
        _require_digest_float(f"skis.{field}", skis[field])
    for side in ("left", "right"):
        ski = skis[side]
        if not isinstance(ski, Mapping):
            raise TypeError(f"skier.skis.{side}: expected mapping")
        _require_exact_keys(f"skis.{side}", ski, (_CS2_ONE_SKI_KEYS,))
        if _require_digest_text(f"skis.{side}.side", ski["side"]) != side:
            raise ValueError(f"skier.skis.{side}.side: expected {side!r}")
        for field in (
            "attack_rad",
            "edge_rad",
            "realized_attack_rad",
            "realized_edge_rad",
            "frame_orientation_residual_rad",
        ):
            _require_digest_float(f"skis.{side}.{field}", ski[field])
        for field in (
            "centerline_origin_world_m",
            "base_origin_world_m",
            "binding_origin_world_m",
            "contact_origin_world_m",
        ):
            _require_f64_digest_array(f"skis.{side}.{field}", ski[field], (3,))
        for field in ("target_F_world_from_ski", "realized_F_world_from_ski"):
            _require_f64_digest_array(f"skis.{side}.{field}", ski[field], (3, 3))
        for field in (
            "analytic_slip_longitudinal_lateral_m_s",
            "realized_slip_longitudinal_lateral_m_s",
        ):
            _require_f64_digest_array(f"skis.{side}.{field}", ski[field], (2,))
    for field in (
        "left_contact_origin_world_m",
        "right_contact_origin_world_m",
    ):
        _require_f64_digest_array(f"contacts.{field}", contacts[field], (3,))
    _require_digest_text("phases.maneuver_id", phases["maneuver_id"])
    _require_digest_float("phases.maneuver_phase", phases["maneuver_phase"])
    clip_ids = phases["animation_clip_ids"]
    if not isinstance(clip_ids, tuple) or not clip_ids:
        raise TypeError("skier.phases.animation_clip_ids: expected non-empty tuple")
    for index, clip_id in enumerate(clip_ids):
        _require_digest_text(f"phases.animation_clip_ids[{index}]", clip_id)
    _require_digest_float("phases.animation_phase", phases["animation_phase"])
    _require_f64_digest_array(
        "phases.animation_blend_weights",
        phases["animation_blend_weights"],
        (len(clip_ids),),
    )


def canonical_skier_digest(
    *,
    root: Mapping[str, object],
    skis: Mapping[str, object],
    contacts: Mapping[str, object],
    phases: Mapping[str, object],
    local_bone_transforms: np.ndarray,
    randomness: Mapping[str, object],
) -> str:
    """Hash only canonical skier/root/pose state, never observation state."""
    payload: dict[str, object] = {
        "schema_version": SKIER_DIGEST_SCHEMA_VERSION,
        "root": root,
        "skis": skis,
        "contacts": contacts,
        "phases": phases,
        "local_bone_transforms": local_bone_transforms,
        "randomness": randomness,
    }
    _reject_forbidden_skier_keys(payload)
    _validate_skier_digest_domains(payload)
    return sha256_canonical(payload)


__all__ = [
    "BRANCH_IDS",
    "CLIP_END_M",
    "CLIP_START_M",
    "COMMAND_DIM",
    "COMMAND_FIELDS",
    "CONTRACT_SCHEMA_VERSION",
    "DEPTH_OF_FIELD_ENABLED",
    "BranchId",
    "BranchProgram",
    "CameraContract",
    "DatasetSplit",
    "ExpectedImageEffect",
    "FIXED_DT_SECONDS",
    "FORWARD_SPEED_MAGNITUDE_M_S",
    "HORIZON_STEPS",
    "LATERAL_SPEED_MAGNITUDE_M_S",
    "LENS_MM",
    "R_CAM_FROM_RIG",
    "R_RIG_FROM_CAM",
    "RENDER_HEIGHT_PX",
    "RENDER_WIDTH_PX",
    "RootSiblingIdentity",
    "SENSOR_WIDTH_MM",
    "SKIER_DIGEST_SCHEMA_VERSION",
    "SKIER_POSE_ROOT_SCHEMA_VERSION",
    "SKIER_ROOT_SCHEMA_VERSION",
    "T_CAM_FROM_RIG_M",
    "T_RIG_FROM_CAM_M",
    "VERTICAL_SPEED_MAGNITUDE_M_S",
    "YAW_RATE_MAGNITUDE_RAD_S",
    "camera_contract_sha256",
    "canonical_branch_programs",
    "canonical_bytes",
    "canonical_skier_digest",
    "default_camera_contract",
    "expected_image_effect",
    "program_by_id",
    "sha256_canonical",
    "sign_eligible",
    "validate_sibling_group",
]
