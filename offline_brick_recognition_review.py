from __future__ import annotations

import argparse
import contextlib
import csv
import ctypes
import io
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np

from Blinx_Public_Class import Blinx_Public_Class
from ImageProcessing.Conversion_3D import Blinx_3D_Conversion
from ImageProcessing.image_recognition import Blinx_image_rec, normalize_angle_90
from brick_process_recording import build_depth_preview, save_rgb_png


DEFAULT_RECORDS_DIR = Path("pic") / "cam1" / "brick_process_records"
GROUND_TRUTH_RECORDS_DIR = Path("pic") / "cam1" / "ground_truth"
PRIMARY_CAPTURE = "primary_pick"
SECONDARY_CAPTURE = "secondary_alignment"
PRIMARY_FILES = (f"{PRIMARY_CAPTURE}_rgb.npy", f"{PRIMARY_CAPTURE}_depth.npy")
SECONDARY_FILES = (f"{SECONDARY_CAPTURE}_rgb.npy", f"{SECONDARY_CAPTURE}_depth.npy")
GROUND_TRUTH_FILES = ("rgb.npy", "depth.npy")
MODEL_PATHS = (
    Path("ImageProcessing") / "gangjin2-m.onnx",
    Path("ImageProcessing") / "hongzhuan-detect.onnx",
    Path("ImageProcessing") / "hongzhuan-seg2.onnx",
)


@dataclass(frozen=True)
class Tolerances:
    px: float = 15.0
    angle_deg: float = 5.0
    depth_mm: float = 10.0
    robot_xy_mm: float = 15.0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay recorded brick-recognition captures offline with visualization and runtime comparison.",
    )
    parser.add_argument("--session-dir", default=None, help="Recorded brick_process_records session directory.")
    parser.add_argument("--case-id", default=None, help="Analyze a single case_xxxx directory only.")
    parser.add_argument(
        "--capture-scope",
        choices=("primary", "secondary", "both"),
        default="both",
        help="Which capture stages to analyze.",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory for offline review results.")
    parser.add_argument("--no-window", action="store_true", help="Skip OpenCV interactive image browsing.")
    parser.add_argument("--conf", type=float, default=None, help="Override segmentation confidence threshold.")
    parser.add_argument("--iou", type=float, default=None, help="Override segmentation IoU threshold.")
    parser.add_argument("--depth-min-valid-mm", type=float, default=None)
    parser.add_argument("--depth-trim-percent", type=float, default=None)
    parser.add_argument("--depth-min-valid-pixels", type=int, default=None)
    parser.add_argument("--depth-min-valid-ratio", type=float, default=None)
    parser.add_argument("--depth-erode-kernel", type=int, default=None)
    parser.add_argument("--depth-erode-iterations", type=int, default=None)
    parser.add_argument("--pca-min-axis-ratio", type=float, default=None)
    parser.add_argument("--px-tol", type=float, default=15.0, help="Pixel tolerance for runtime_consistent.")
    parser.add_argument("--angle-tol-deg", type=float, default=5.0, help="Angle tolerance in degrees.")
    parser.add_argument("--depth-tol-mm", type=float, default=10.0, help="Depth tolerance in millimeters.")
    parser.add_argument("--robot-xy-tol-mm", type=float, default=15.0, help="Robot XY tolerance in millimeters.")
    return parser.parse_args(argv)


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_model_files() -> None:
    missing = [str(path) for path in MODEL_PATHS if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing ONNX model file(s): {joined}")


def _list_session_dirs(base_dir: Path | str) -> List[Path]:
    base_dir = Path(base_dir).resolve()
    if not base_dir.exists():
        return []

    sessions: list[Path] = []
    for path in base_dir.iterdir():
        if path.is_dir():
            sessions.append(path)
    return sorted(sessions, key=lambda item: item.name)


def _session_has_case_dirs(session_dir: Path) -> bool:
    return any(path.is_dir() and path.name.startswith("case_") for path in session_dir.iterdir())


def discover_latest_session_dir(base_dir: Optional[Path | str] = None) -> Path:
    if base_dir is not None:
        sessions = _list_session_dirs(base_dir)
        if not sessions:
            raise FileNotFoundError(f"No session directories found under: {Path(base_dir).resolve()}")
        for session_dir in reversed(sessions):
            if _session_has_case_dirs(session_dir):
                return session_dir
        raise FileNotFoundError(
            f"No session directories with case_* children found under: {Path(base_dir).resolve()}"
        )

    for candidate_base in (DEFAULT_RECORDS_DIR, GROUND_TRUTH_RECORDS_DIR):
        sessions = _list_session_dirs(candidate_base)
        for session_dir in reversed(sessions):
            if _session_has_case_dirs(session_dir):
                return session_dir
    searched = f"{GROUND_TRUTH_RECORDS_DIR.resolve()} , {DEFAULT_RECORDS_DIR.resolve()}"
    raise FileNotFoundError(f"No session directories found under: {searched}")


def detect_case_format(case_dir: Path) -> str:
    if (case_dir / PRIMARY_FILES[0]).exists() and (case_dir / PRIMARY_FILES[1]).exists():
        return "brick_process_record"
    if (case_dir / GROUND_TRUTH_FILES[0]).exists() and (case_dir / GROUND_TRUTH_FILES[1]).exists():
        return "ground_truth"
    return "unknown"


def load_primary_arrays(case_dir: Path) -> Optional[tuple[np.ndarray, np.ndarray]]:
    case_format = detect_case_format(case_dir)
    if case_format == "brick_process_record":
        rgb_path = case_dir / PRIMARY_FILES[0]
        depth_path = case_dir / PRIMARY_FILES[1]
    elif case_format == "ground_truth":
        rgb_path = case_dir / GROUND_TRUTH_FILES[0]
        depth_path = case_dir / GROUND_TRUTH_FILES[1]
    else:
        return None

    rgb = np.load(rgb_path, allow_pickle=False)
    depth = np.load(depth_path, allow_pickle=False)
    return rgb, depth


def resolve_session_dir(session_dir: Optional[str]) -> Path:
    if session_dir:
        path = Path(session_dir).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Session directory does not exist: {path}")
        return path
    return discover_latest_session_dir()


def resolve_output_dir(session_dir: Path, output_dir: Optional[str]) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    return (session_dir / "offline_review" / now_timestamp()).resolve()


def list_case_dirs(session_dir: Path, case_id: Optional[str] = None) -> List[Path]:
    if case_id:
        case_dir = (session_dir / case_id).resolve()
        if not case_dir.exists():
            raise FileNotFoundError(f"Case directory does not exist: {case_dir}")
        return [case_dir]

    case_dirs = [path for path in session_dir.iterdir() if path.is_dir() and path.name.startswith("case_")]
    if not case_dirs:
        raise FileNotFoundError(f"No case_* directories found under: {session_dir}")
    return sorted(case_dirs, key=lambda item: item.name)


def capture_scopes(scope: str) -> List[str]:
    if scope == "primary":
        return [PRIMARY_CAPTURE]
    if scope == "secondary":
        return [SECONDARY_CAPTURE]
    return [PRIMARY_CAPTURE, SECONDARY_CAPTURE]


def load_metadata(case_dir: Path) -> Dict[str, Any]:
    metadata_path = case_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json for case: {case_dir}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_capture_arrays(case_dir: Path, capture_stage: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if capture_stage == PRIMARY_CAPTURE:
        return load_primary_arrays(case_dir)
    elif capture_stage == SECONDARY_CAPTURE:
        rgb_name, depth_name = SECONDARY_FILES
    else:
        raise ValueError(f"Unsupported capture stage: {capture_stage}")

    rgb_path = case_dir / rgb_name
    depth_path = case_dir / depth_name
    if not rgb_path.exists() or not depth_path.exists():
        return None

    rgb = np.load(rgb_path, allow_pickle=False)
    depth = np.load(depth_path, allow_pickle=False)
    return rgb, depth


def extract_primary_runtime_reference(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    capture = metadata.get("captures", {}).get(PRIMARY_CAPTURE, {})
    pick_result = (capture.get("extra") or {}).get("pick_result")
    if not isinstance(pick_result, dict):
        return None
    return {
        "pixel_x": pick_result.get("pixel_x"),
        "pixel_y": pick_result.get("pixel_y"),
        "depth_mm": pick_result.get("depth_mm"),
        "angle_deg": pick_result.get("angle_deg"),
        "score": pick_result.get("score"),
    }


def extract_secondary_runtime_reference(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    capture = metadata.get("captures", {}).get(SECONDARY_CAPTURE, {})
    alignment_result = (capture.get("extra") or {}).get("alignment_result")
    if not isinstance(alignment_result, (list, tuple)) or len(alignment_result) < 2:
        return None
    return {
        "pixel_x": float(alignment_result[0]),
        "pixel_y": float(alignment_result[1]),
        "depth_mm": None,
        "angle_deg": None,
        "score": None,
    }


def extract_manual_ground_truth(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    capture_tcp_pose = metadata.get("capture_tcp_pose")
    ground_truth_tcp_pose = metadata.get("ground_truth_tcp_pose")
    capture_joint_pos = metadata.get("capture_joint_pos")
    ground_truth_joint_pos = metadata.get("ground_truth_joint_pos")
    if not all(isinstance(item, (list, tuple)) and len(item) >= 6 for item in (
        capture_tcp_pose,
        ground_truth_tcp_pose,
        capture_joint_pos,
        ground_truth_joint_pos,
    )):
        return None
    return {
        "capture_tcp_pose": [float(value) for value in capture_tcp_pose[:6]],
        "ground_truth_tcp_pose": [float(value) for value in ground_truth_tcp_pose[:6]],
        "capture_joint_pos": [float(value) for value in capture_joint_pos[:6]],
        "ground_truth_joint_pos": [float(value) for value in ground_truth_joint_pos[:6]],
    }


def apply_recognizer_overrides(recognizer: Any, args: argparse.Namespace) -> Any:
    if args.conf is not None:
        recognizer.conf = float(args.conf)
    if args.iou is not None:
        recognizer.iou = float(args.iou)

    cfg_updates = {
        "depth_min_valid_mm": args.depth_min_valid_mm,
        "depth_trim_percent": args.depth_trim_percent,
        "depth_min_valid_pixels": args.depth_min_valid_pixels,
        "depth_min_valid_ratio": args.depth_min_valid_ratio,
        "depth_erode_kernel": args.depth_erode_kernel,
        "depth_erode_iterations": args.depth_erode_iterations,
        "pca_min_axis_ratio": args.pca_min_axis_ratio,
    }
    for key, value in cfg_updates.items():
        if value is not None:
            recognizer.rgbd_cfg[key] = value
    return recognizer


def create_recognizer(args: argparse.Namespace) -> Any:
    validate_model_files()
    public_class = Blinx_Public_Class()
    recognizer = Blinx_image_rec(public_class)
    return apply_recognizer_overrides(recognizer, args)


def _apply_mask_overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    result = image.copy()
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return result
    overlay = np.zeros_like(result)
    overlay[mask_bool] = np.array(color, dtype=np.uint8)
    blended = cv2.addWeighted(result, 1.0, overlay, alpha, 0.0)
    result[mask_bool] = blended[mask_bool]
    return result


def _draw_cross(image: np.ndarray, center: tuple[int, int], color: tuple[int, int, int], size: int = 12) -> None:
    cv2.line(image, (center[0] - size, center[1]), (center[0] + size, center[1]), color, 2, cv2.LINE_AA)
    cv2.line(image, (center[0], center[1] - size), (center[0], center[1] + size), color, 2, cv2.LINE_AA)


def _draw_angle_stub(
    image: np.ndarray,
    center: tuple[int, int],
    angle_deg: Optional[float],
    color: tuple[int, int, int],
    length: int = 72,
) -> None:
    if angle_deg is None:
        return
    angle_rad = np.deg2rad(float(angle_deg))
    end_point = (
        int(round(center[0] + np.cos(angle_rad) * length)),
        int(round(center[1] + np.sin(angle_rad) * length)),
    )
    cv2.line(image, center, end_point, color, 2, cv2.LINE_AA)


def _draw_label(
    image: np.ndarray,
    anchor: tuple[int, int],
    lines: Iterable[str],
    color: tuple[int, int, int],
) -> None:
    lines = [line for line in lines if line]
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2
    line_height = 22
    x = max(8, min(int(anchor[0]), image.shape[1] - 320))
    y = max(24, min(int(anchor[1]), image.shape[0] - line_height * len(lines) - 8))
    for index, line in enumerate(lines):
        origin = (x, y + index * line_height)
        cv2.putText(image, line, origin, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(image, line, origin, font, scale, color, thickness, cv2.LINE_AA)


def _compute_rotated_box(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if xs.size < 2:
        return None
    points = np.column_stack((xs, ys)).astype(np.float32)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    return np.intp(np.round(box))


def _pad_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.shape[0] == target_height:
        return image
    pad_bottom = target_height - image.shape[0]
    return cv2.copyMakeBorder(
        image,
        0,
        pad_bottom,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def _pad_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    if image.shape[1] == target_width:
        return image
    pad_right = target_width - image.shape[1]
    return cv2.copyMakeBorder(
        image,
        0,
        0,
        0,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def build_analysis_panel(display_rgb: np.ndarray, depth_preview: np.ndarray, header_lines: Sequence[str]) -> np.ndarray:
    target_height = max(display_rgb.shape[0], depth_preview.shape[0])
    left = _pad_to_height(display_rgb, target_height)
    right = _pad_to_height(depth_preview, target_height)
    separator = np.full((target_height, 16, 3), 255, dtype=np.uint8)
    content = np.hstack((left, separator, right))

    line_height = 24
    header_height = 20 + line_height * max(len(header_lines), 1)
    panel = np.full((header_height + content.shape[0], content.shape[1], 3), 255, dtype=np.uint8)
    panel[header_height:, :, :] = content

    for index, line in enumerate(header_lines):
        origin = (16, 26 + index * line_height)
        cv2.putText(panel, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0, 0, 0), 2, cv2.LINE_AA)
    return panel


def combine_primary_and_depth_analysis(primary_panel: np.ndarray, depth_panel: np.ndarray) -> np.ndarray:
    target_width = max(primary_panel.shape[1], depth_panel.shape[1])
    top = _pad_to_width(primary_panel, target_width)
    bottom = _pad_to_width(depth_panel, target_width)
    separator = np.full((18, target_width, 3), 255, dtype=np.uint8)
    return np.vstack((top, separator, bottom))


def load_bgr_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"Failed to load image for browsing: {path}")
    return image


def get_screen_size() -> tuple[int, int]:
    try:
        user32 = ctypes.windll.user32
        try:
            user32.SetProcessDPIAware()
        except AttributeError:
            pass
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    return 1600, 900


def fit_image_to_screen(
    image: np.ndarray,
    *,
    screen_size: Optional[tuple[int, int]] = None,
    width_ratio: float = 0.92,
    height_ratio: float = 0.86,
) -> tuple[np.ndarray, float]:
    if screen_size is None:
        screen_size = get_screen_size()

    screen_width, screen_height = screen_size
    max_width = max(int(screen_width * width_ratio), 1)
    max_height = max(int(screen_height * height_ratio), 1)

    height, width = image.shape[:2]
    scale = min(max_width / float(width), max_height / float(height), 1.0)
    if scale >= 0.999:
        return image, 1.0

    resized = cv2.resize(
        image,
        (max(int(round(width * scale)), 1), max(int(round(height * scale)), 1)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def sanitize_primary_candidate(
    candidate: Optional[Dict[str, Any]],
    *,
    rank_index: Optional[int] = None,
    is_selected: bool = False,
) -> Optional[Dict[str, Any]]:
    if candidate is None:
        return None
    return {
        "rank_index": rank_index,
        "is_selected": is_selected,
        "class_id": candidate.get("class_id"),
        "score": candidate.get("score"),
        "bbox": to_serializable(candidate.get("bbox")),
        "pixel_x": candidate.get("pixel_x"),
        "pixel_y": candidate.get("pixel_y"),
        "depth_mm": candidate.get("depth_mm"),
        "angle_deg": candidate.get("angle_deg"),
        "raw_pca_angle_deg": candidate.get("raw_pca_angle_deg"),
        "valid_depth_count": candidate.get("valid_depth_count"),
        "valid_depth_ratio": candidate.get("valid_depth_ratio"),
        "mask_foreground_count": candidate.get("mask_foreground_count"),
        "axis_ratio": candidate.get("axis_ratio"),
        "angle_fallback": candidate.get("angle_fallback"),
        "is_valid": candidate.get("is_valid"),
    }


def sanitize_primary_candidates(
    candidates: Sequence[Dict[str, Any]],
    selected_candidate: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        sanitized_candidate = sanitize_primary_candidate(
            candidate,
            rank_index=index,
            is_selected=(candidate is selected_candidate),
        )
        if sanitized_candidate is not None:
            sanitized.append(sanitized_candidate)
    return sanitized


def sanitize_secondary_result(result: Optional[Sequence[float]]) -> Optional[Dict[str, Any]]:
    if not isinstance(result, (list, tuple)) or len(result) < 2:
        return None
    return {
        "pixel_x": float(result[0]),
        "pixel_y": float(result[1]),
        "depth_mm": None,
        "angle_deg": None,
        "score": None,
    }


def sanitize_depth_candidate(
    candidate: Optional[Dict[str, Any]],
    *,
    rank_index: Optional[int] = None,
    is_selected: bool = False,
) -> Optional[Dict[str, Any]]:
    if candidate is None:
        return None
    return {
        "rank_index": rank_index,
        "is_selected": is_selected,
        "source": candidate.get("source"),
        "pixel_x": candidate.get("pixel_x"),
        "pixel_y": candidate.get("pixel_y"),
        "depth_mm": candidate.get("depth_mm"),
        "angle_deg": candidate.get("angle_deg"),
        "score": candidate.get("score"),
        "geometry_score": candidate.get("geometry_score"),
        "bbox": to_serializable(candidate.get("bbox")),
        "box_points": to_serializable(candidate.get("box_points")),
        "long_side_mm": candidate.get("long_side_mm"),
        "short_side_mm": candidate.get("short_side_mm"),
        "median_height_mm": candidate.get("median_height_mm"),
        "height_std_mm": candidate.get("height_std_mm"),
        "size_match_score": candidate.get("size_match_score"),
        "planarity_score": candidate.get("planarity_score"),
        "rectangularity_score": candidate.get("rectangularity_score"),
        "completeness_score": candidate.get("completeness_score"),
        "height_consistency_score": candidate.get("height_consistency_score"),
        "touch_border": candidate.get("touch_border"),
        "valid_depth_count": candidate.get("valid_depth_count"),
        "valid_depth_ratio": candidate.get("valid_depth_ratio"),
        "axis_ratio": candidate.get("axis_ratio"),
        "camera_center_xyz": to_serializable(candidate.get("camera_center_xyz")),
        "is_valid": candidate.get("is_valid"),
    }


def sanitize_depth_candidates(
    candidates: Sequence[Dict[str, Any]],
    selected_candidate: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        sanitized_candidate = sanitize_depth_candidate(
            candidate,
            rank_index=index,
            is_selected=(candidate is selected_candidate),
        )
        if sanitized_candidate is not None:
            sanitized.append(sanitized_candidate)
    return sanitized


def run_primary_review_inference(recognizer: Any, rgb: np.ndarray, depth: np.ndarray) -> Dict[str, Any]:
    debug_method = getattr(recognizer, "blinx_brick_image_rec_debug", None)
    if callable(debug_method):
        report = debug_method(rgb, depth)
        ranked_candidates = list(report.get("ranked_candidates") or report.get("candidates") or [])
        return {
            "selected_candidate": report.get("selected_candidate"),
            "ranked_candidates": ranked_candidates,
            "selected_rank_index": report.get("selected_rank_index"),
        }

    _, _, candidate = recognizer.blinx_brick_image_rec2(rgb, depth)
    ranked_candidates = [] if candidate is None else [candidate]
    return {
        "selected_candidate": candidate,
        "ranked_candidates": ranked_candidates,
        "selected_rank_index": 0 if candidate is not None else None,
    }


def run_depth_review_inference(recognizer: Any, rgb: np.ndarray, depth: np.ndarray) -> Dict[str, Any]:
    depth_method = getattr(recognizer, "blinx_brick_depth_candidates", None)
    if not callable(depth_method):
        return {
            "rgb_debug": rgb.copy(),
            "depth_debug": build_depth_preview(depth),
            "debug_images": {},
            "candidates": [],
            "ranked_candidates": [],
            "selected_candidate": None,
            "selected_rank_index": None,
            "board_depth_mm": None,
            "workspace_roi": None,
        }
    report = depth_method(rgb, depth)
    return {
        "rgb_debug": report.get("rgb_debug", rgb.copy()),
        "depth_debug": report.get("depth_debug", build_depth_preview(depth)),
        "debug_images": dict(report.get("debug_images") or {}),
        "candidates": list(report.get("candidates") or []),
        "ranked_candidates": list(report.get("ranked_candidates") or []),
        "selected_candidate": report.get("selected_candidate"),
        "selected_rank_index": report.get("selected_rank_index"),
        "board_depth_mm": report.get("board_depth_mm"),
        "workspace_roi": report.get("workspace_roi"),
    }


def run_secondary_depth_review_inference(recognizer: Any, rgb: np.ndarray, depth: np.ndarray) -> Dict[str, Any]:
    depth_method = getattr(recognizer, "blinx_brick_secondary_depth_candidates", None)
    if callable(depth_method):
        report = depth_method(rgb, depth)
        return {
            "rgb_debug": report.get("rgb_debug", rgb.copy()),
            "depth_debug": report.get("depth_debug", build_depth_preview(depth)),
            "debug_images": dict(report.get("debug_images") or {}),
            "candidates": list(report.get("candidates") or []),
            "ranked_candidates": list(report.get("ranked_candidates") or []),
            "selected_candidate": report.get("selected_candidate"),
            "selected_rank_index": report.get("selected_rank_index"),
            "board_depth_mm": report.get("board_depth_mm"),
            "workspace_roi": report.get("workspace_roi"),
        }
    return run_depth_review_inference(recognizer, rgb, depth)


def compute_primary_status(
    offline_result: Optional[Dict[str, Any]],
    runtime_reference: Optional[Dict[str, Any]],
    tolerances: Tolerances,
) -> tuple[str, Dict[str, Optional[float]]]:
    delta = {
        "delta_x_px": None,
        "delta_y_px": None,
        "delta_angle_deg": None,
        "delta_depth_mm": None,
    }
    if offline_result is None and runtime_reference is None:
        return "both_missing", delta
    if offline_result is None:
        return "offline_missing", delta
    if runtime_reference is None:
        return "runtime_missing", delta

    delta["delta_x_px"] = float(offline_result["pixel_x"]) - float(runtime_reference["pixel_x"])
    delta["delta_y_px"] = float(offline_result["pixel_y"]) - float(runtime_reference["pixel_y"])
    if offline_result.get("angle_deg") is not None and runtime_reference.get("angle_deg") is not None:
        delta["delta_angle_deg"] = float(
            normalize_angle_90(float(offline_result["angle_deg"]) - float(runtime_reference["angle_deg"]))
        )
    if offline_result.get("depth_mm") is not None and runtime_reference.get("depth_mm") is not None:
        delta["delta_depth_mm"] = float(offline_result["depth_mm"]) - float(runtime_reference["depth_mm"])

    is_consistent = (
        abs(delta["delta_x_px"]) <= tolerances.px
        and abs(delta["delta_y_px"]) <= tolerances.px
        and (delta["delta_angle_deg"] is None or abs(delta["delta_angle_deg"]) <= tolerances.angle_deg)
        and (delta["delta_depth_mm"] is None or abs(delta["delta_depth_mm"]) <= tolerances.depth_mm)
    )
    return ("runtime_consistent" if is_consistent else "review_needed"), delta


def compute_secondary_status(
    offline_result: Optional[Dict[str, Any]],
    runtime_reference: Optional[Dict[str, Any]],
    tolerances: Tolerances,
) -> tuple[str, Dict[str, Optional[float]]]:
    delta = {
        "delta_x_px": None,
        "delta_y_px": None,
        "delta_angle_deg": None,
        "delta_depth_mm": None,
    }
    if offline_result is None and runtime_reference is None:
        return "both_missing", delta
    if offline_result is None:
        return "offline_missing", delta
    if runtime_reference is None:
        return "runtime_missing", delta

    delta["delta_x_px"] = float(offline_result["pixel_x"]) - float(runtime_reference["pixel_x"])
    delta["delta_y_px"] = float(offline_result["pixel_y"]) - float(runtime_reference["pixel_y"])
    is_consistent = abs(delta["delta_x_px"]) <= tolerances.px and abs(delta["delta_y_px"]) <= tolerances.px
    return ("runtime_consistent" if is_consistent else "review_needed"), delta


def compute_depth_runtime_proxy_metrics(
    offline_result: Optional[Dict[str, Any]],
    runtime_reference: Optional[Dict[str, Any]],
    tolerances: Tolerances,
) -> Dict[str, Any]:
    status, delta = compute_primary_status(offline_result, runtime_reference, tolerances)
    return {
        "status": status,
        "runtime_reference": runtime_reference,
        "delta": delta,
    }


def compute_depth_manual_ground_truth_metrics(
    candidate: Optional[Dict[str, Any]],
    metadata: Dict[str, Any],
    recognizer: Any,
    tolerances: Tolerances,
) -> Dict[str, Any]:
    manual_gt = extract_manual_ground_truth(metadata)
    if manual_gt is None:
        return {
            "status": "gt_missing",
            "robot_x_mm": None,
            "robot_y_mm": None,
            "target_z_mm": None,
            "expected_j6_delta_deg": None,
            "delta_robot_x_mm": None,
            "delta_robot_y_mm": None,
            "delta_pick_z_mm": None,
            "delta_j6_delta_deg": None,
        }

    if candidate is None:
        return {
            "status": "depth_missing",
            "robot_x_mm": None,
            "robot_y_mm": None,
            "target_z_mm": None,
            "expected_j6_delta_deg": float(
                manual_gt["ground_truth_joint_pos"][5] - manual_gt["capture_joint_pos"][5]
            ),
            "delta_robot_x_mm": None,
            "delta_robot_y_mm": None,
            "delta_pick_z_mm": None,
            "delta_j6_delta_deg": None,
            "manual_ground_truth": manual_gt,
        }

    conversion = Blinx_3D_Conversion()
    with contextlib.redirect_stdout(io.StringIO()):
        cam_x, cam_y, cam_z = recognizer.image_camera.blinx_image_to_camera(
            candidate["pixel_x"],
            candidate["pixel_y"],
            candidate["depth_mm"],
        )
        robot_xy = conversion.blinx_conversion([cam_x, cam_y, cam_z])

    robot_x_mm = float(robot_xy["X"])
    robot_y_mm = float(robot_xy["Y"])
    target_z_mm = float(manual_gt["capture_tcp_pose"][2] - float(candidate["depth_mm"]) + 166.0)
    expected_j6_delta_deg = float(manual_gt["ground_truth_joint_pos"][5] - manual_gt["capture_joint_pos"][5])
    delta_robot_x_mm = float(robot_x_mm - manual_gt["ground_truth_tcp_pose"][0])
    delta_robot_y_mm = float(robot_y_mm - manual_gt["ground_truth_tcp_pose"][1])
    delta_pick_z_mm = float(target_z_mm - manual_gt["ground_truth_tcp_pose"][2])
    delta_j6_delta_deg = float(normalize_angle_90(float(candidate["angle_deg"]) - expected_j6_delta_deg))
    is_consistent = (
        abs(delta_robot_x_mm) <= tolerances.robot_xy_mm
        and abs(delta_robot_y_mm) <= tolerances.robot_xy_mm
        and abs(delta_pick_z_mm) <= tolerances.depth_mm
        and abs(delta_j6_delta_deg) <= tolerances.angle_deg
    )
    return {
        "status": "gt_consistent" if is_consistent else "gt_review_needed",
        "robot_x_mm": robot_x_mm,
        "robot_y_mm": robot_y_mm,
        "target_z_mm": target_z_mm,
        "expected_j6_delta_deg": expected_j6_delta_deg,
        "delta_robot_x_mm": delta_robot_x_mm,
        "delta_robot_y_mm": delta_robot_y_mm,
        "delta_pick_z_mm": delta_pick_z_mm,
        "delta_j6_delta_deg": delta_j6_delta_deg,
        "manual_ground_truth": manual_gt,
    }


def _format_optional_float(value: Optional[float], pattern: str) -> str:
    if value is None:
        return "None"
    return pattern.format(float(value))


def _draw_primary_candidate(
    image: np.ndarray,
    candidate: Dict[str, Any],
    *,
    mask_color: tuple[int, int, int],
    mask_alpha: float,
    box_color: tuple[int, int, int],
    center_color: tuple[int, int, int],
    label_color: tuple[int, int, int],
    label_prefix: str,
) -> None:
    image[:] = _apply_mask_overlay(image, candidate["mask"], mask_color, mask_alpha)
    box = _compute_rotated_box(candidate["mask"])
    if box is not None:
        cv2.drawContours(image, [box], 0, box_color, 2, cv2.LINE_AA)
    if candidate.get("pixel_x") is None or candidate.get("pixel_y") is None:
        return

    center = (int(candidate["pixel_x"]), int(candidate["pixel_y"]))
    cv2.circle(image, center, 6, center_color, -1, cv2.LINE_AA)
    _draw_label(
        image,
        (center[0] + 18, center[1] - 12),
        [
            f"{label_prefix} ({center[0]}, {center[1]})",
            f"Angle {_format_optional_float(candidate.get('angle_deg'), '{:.2f}')} deg",
            (
                f"Depth {_format_optional_float(candidate.get('depth_mm'), '{:.1f}')} mm  "
                f"Score {_format_optional_float(candidate.get('score'), '{:.3f}')}"
            ),
        ],
        label_color,
    )


def build_primary_candidate_lines(candidates: Sequence[Dict[str, Any]]) -> List[str]:
    if not candidates:
        return ["Offline candidates: 0"]

    lines = [f"Offline candidates: {len(candidates)}"]
    for candidate in candidates:
        prefix = (
            f"Selected #{int(candidate['rank_index']) + 1}"
            if candidate.get("is_selected")
            else f"Cand #{int(candidate['rank_index']) + 1}"
        )
        lines.append(
            f"{prefix}: ({candidate['pixel_x']}, {candidate['pixel_y']}) "
            f"A {_format_optional_float(candidate.get('angle_deg'), '{:.2f}')} "
            f"Z {_format_optional_float(candidate.get('depth_mm'), '{:.1f}')} "
            f"S {_format_optional_float(candidate.get('score'), '{:.3f}')}"
        )
    return lines


def build_primary_analysis_image(
    rgb_image: np.ndarray,
    depth_map: np.ndarray,
    offline_result: Optional[Dict[str, Any]],
    offline_candidates: Sequence[Dict[str, Any]],
    selected_rank_index: Optional[int],
    runtime_reference: Optional[Dict[str, Any]],
    case_id: str,
    status: str,
    delta: Dict[str, Optional[float]],
) -> np.ndarray:
    display = np.ascontiguousarray(rgb_image.copy())

    for index, candidate in enumerate(offline_candidates):
        if selected_rank_index is not None and index == selected_rank_index:
            continue
        _draw_primary_candidate(
            display,
            candidate,
            mask_color=(200, 220, 255),
            mask_alpha=0.16,
            box_color=(255, 165, 0),
            center_color=(255, 165, 0),
            label_color=(255, 210, 120),
            label_prefix=f"Cand #{index + 1}",
        )

    if offline_result is not None:
        selected_label = f"Selected #{selected_rank_index + 1}" if selected_rank_index is not None else "Selected"
        _draw_primary_candidate(
            display,
            offline_result,
            mask_color=(173, 216, 230),
            mask_alpha=0.38,
            box_color=(0, 255, 0),
            center_color=(255, 0, 0),
            label_color=(255, 255, 0),
            label_prefix=selected_label,
        )

    if runtime_reference is not None and runtime_reference.get("pixel_x") is not None and runtime_reference.get("pixel_y") is not None:
        runtime_center = (int(round(float(runtime_reference["pixel_x"]))), int(round(float(runtime_reference["pixel_y"]))))
        _draw_cross(display, runtime_center, (255, 0, 255), size=12)
        _draw_angle_stub(display, runtime_center, runtime_reference.get("angle_deg"), (255, 0, 255), length=70)
        _draw_label(
            display,
            (runtime_center[0] + 18, runtime_center[1] + 34),
            [
                f"Runtime ({runtime_center[0]}, {runtime_center[1]})",
                (
                    f"dX {delta['delta_x_px']:.1f}  dY {delta['delta_y_px']:.1f}"
                    if delta["delta_x_px"] is not None and delta["delta_y_px"] is not None
                    else ""
                ),
                (
                    f"dAngle {delta['delta_angle_deg']:.2f}  dDepth {delta['delta_depth_mm']:.1f}"
                    if delta["delta_angle_deg"] is not None and delta["delta_depth_mm"] is not None
                    else f"dAngle {delta['delta_angle_deg']:.2f}" if delta["delta_angle_deg"] is not None else ""
                ),
            ],
            (255, 0, 255),
        )

    header_lines = [
        f"Case: {case_id}  Stage: {PRIMARY_CAPTURE}  Status: {status}",
        "Offline result is the replayed recognition. Runtime mark is a historical reference only.",
        (
            f"Delta X {delta['delta_x_px']:.1f} px  Delta Y {delta['delta_y_px']:.1f} px  "
            f"Delta Angle {delta['delta_angle_deg']:.2f} deg  Delta Depth {delta['delta_depth_mm']:.1f} mm"
            if (
                delta["delta_x_px"] is not None
                and delta["delta_y_px"] is not None
                and delta["delta_angle_deg"] is not None
                and delta["delta_depth_mm"] is not None
            )
            else "No runtime comparison available for at least one metric."
        ),
    ]
    header_lines.extend(build_primary_candidate_lines(sanitize_primary_candidates(offline_candidates, offline_result)))
    return build_analysis_panel(display, build_depth_preview(depth_map), header_lines)


def build_depth_analysis_image(
    panel_image: np.ndarray,
    *,
    case_id: str,
    board_depth_mm: Optional[float],
    depth_status: str,
    manual_gt_metrics: Dict[str, Any],
    runtime_proxy_metrics: Dict[str, Any],
    selected_candidate: Optional[Dict[str, Any]],
) -> np.ndarray:
    header_lines = [
        f"Case: {case_id}  Stage: {PRIMARY_CAPTURE} depth_geometry  Status: {depth_status}",
        (
            f"Board {board_depth_mm:.1f} mm  "
            f"Sel ({selected_candidate['pixel_x']}, {selected_candidate['pixel_y']})  "
            f"Z {selected_candidate['depth_mm']:.1f}  A {selected_candidate['angle_deg']:.2f}"
            if board_depth_mm is not None and selected_candidate is not None
            else "No valid depth-selected candidate."
        ),
        (
            f"dRobotX {manual_gt_metrics['delta_robot_x_mm']:.1f}  "
            f"dRobotY {manual_gt_metrics['delta_robot_y_mm']:.1f}  "
            f"dPickZ {manual_gt_metrics['delta_pick_z_mm']:.1f}  "
            f"dJ6 {manual_gt_metrics['delta_j6_delta_deg']:.2f}"
            if manual_gt_metrics.get("delta_robot_x_mm") is not None
            else f"Ground truth status: {manual_gt_metrics['status']}"
        ),
        (
            f"Runtime proxy: {runtime_proxy_metrics['status']}  "
            f"dX {_format_optional_float(runtime_proxy_metrics['delta'].get('delta_x_px'), '{:.1f}')}  "
            f"dY {_format_optional_float(runtime_proxy_metrics['delta'].get('delta_y_px'), '{:.1f}')}  "
            f"dA {_format_optional_float(runtime_proxy_metrics['delta'].get('delta_angle_deg'), '{:.2f}')}  "
            f"dZ {_format_optional_float(runtime_proxy_metrics['delta'].get('delta_depth_mm'), '{:.1f}')}"
        ),
    ]
    line_height = 24
    header_height = 20 + line_height * len(header_lines)
    result = np.full((header_height + panel_image.shape[0], panel_image.shape[1], 3), 255, dtype=np.uint8)
    result[header_height:, :, :] = panel_image
    for index, line in enumerate(header_lines):
        origin = (16, 26 + index * line_height)
        cv2.putText(result, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0, 0, 0), 2, cv2.LINE_AA)
    return result


def build_secondary_analysis_image(
    display_rgb: np.ndarray,
    depth_map: np.ndarray,
    offline_result: Optional[Dict[str, Any]],
    runtime_reference: Optional[Dict[str, Any]],
    case_id: str,
    status: str,
    delta: Dict[str, Optional[float]],
) -> np.ndarray:
    display = np.ascontiguousarray(display_rgb.copy())

    if offline_result is not None:
        center = (int(round(float(offline_result["pixel_x"]))), int(round(float(offline_result["pixel_y"]))))
        cv2.circle(display, center, 6, (255, 0, 0), -1, cv2.LINE_AA)
        _draw_label(
            display,
            (center[0] + 18, center[1] - 12),
            [f"Center ({center[0]}, {center[1]})"],
            (255, 255, 0),
        )

    if runtime_reference is not None and runtime_reference.get("pixel_x") is not None and runtime_reference.get("pixel_y") is not None:
        runtime_center = (int(round(float(runtime_reference["pixel_x"]))), int(round(float(runtime_reference["pixel_y"]))))
        _draw_cross(display, runtime_center, (255, 0, 255), size=12)
        _draw_label(
            display,
            (runtime_center[0] + 18, runtime_center[1] + 34),
            [
                f"Runtime ({runtime_center[0]}, {runtime_center[1]})",
                (
                    f"dX {delta['delta_x_px']:.1f}  dY {delta['delta_y_px']:.1f}"
                    if delta["delta_x_px"] is not None and delta["delta_y_px"] is not None
                    else ""
                ),
            ],
            (255, 0, 255),
        )

    header_lines = [
        f"Case: {case_id}  Stage: {SECONDARY_CAPTURE}  Status: {status}",
        "Secondary alignment RGB reference replay. Depth geometry is shown in the paired depth panel.",
        (
            f"Delta X {delta['delta_x_px']:.1f} px  Delta Y {delta['delta_y_px']:.1f} px"
            if delta["delta_x_px"] is not None and delta["delta_y_px"] is not None
            else "No runtime comparison available."
        ),
    ]
    return build_analysis_panel(display, build_depth_preview(depth_map), header_lines)


def build_secondary_depth_analysis_image(
    panel_image: np.ndarray,
    *,
    case_id: str,
    board_depth_mm: Optional[float],
    depth_status: str,
    runtime_proxy_metrics: Dict[str, Any],
    selected_candidate: Optional[Dict[str, Any]],
) -> np.ndarray:
    delta = dict(runtime_proxy_metrics.get("delta") or {})
    header_lines = [
        f"Case: {case_id}  Stage: {SECONDARY_CAPTURE}.depth_geometry  Status: {depth_status}",
        (
            f"Board {board_depth_mm:.1f} mm  "
            f"Sel ({selected_candidate['pixel_x']}, {selected_candidate['pixel_y']})  "
            f"Z {selected_candidate['depth_mm']:.1f}  A {selected_candidate['angle_deg']:.2f}"
            if board_depth_mm is not None and selected_candidate is not None
            else "No valid secondary depth-selected candidate."
        ),
        (
            f"Runtime proxy: {runtime_proxy_metrics['status']}  "
            f"dX {_format_optional_float(delta.get('delta_x_px'), '{:.1f}')}  "
            f"dY {_format_optional_float(delta.get('delta_y_px'), '{:.1f}')}"
        ),
    ]
    line_height = 24
    header_height = 20 + line_height * len(header_lines)
    result = np.full((header_height + panel_image.shape[0], panel_image.shape[1], 3), 255, dtype=np.uint8)
    result[header_height:, :, :] = panel_image
    for index, line in enumerate(header_lines):
        origin = (16, 26 + index * line_height)
        cv2.putText(result, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0, 0, 0), 2, cv2.LINE_AA)
    return result


def build_stage_row(
    *,
    case_id: str,
    capture_stage: str,
    status: str,
    offline_result: Optional[Dict[str, Any]],
    offline_candidate_count: Optional[int],
    offline_selected_rank: Optional[int],
    runtime_reference: Optional[Dict[str, Any]],
    delta: Dict[str, Optional[float]],
    analysis_image_path: Path,
) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "capture_stage": capture_stage,
        "offline_detected": offline_result is not None,
        "runtime_reference_detected": runtime_reference is not None,
        "status": status,
        "offline_candidate_count": offline_candidate_count,
        "offline_selected_rank": offline_selected_rank,
        "offline_pixel_x": None if offline_result is None else offline_result.get("pixel_x"),
        "offline_pixel_y": None if offline_result is None else offline_result.get("pixel_y"),
        "offline_depth_mm": None if offline_result is None else offline_result.get("depth_mm"),
        "offline_angle_deg": None if offline_result is None else offline_result.get("angle_deg"),
        "offline_score": None if offline_result is None else offline_result.get("score"),
        "runtime_pixel_x": None if runtime_reference is None else runtime_reference.get("pixel_x"),
        "runtime_pixel_y": None if runtime_reference is None else runtime_reference.get("pixel_y"),
        "runtime_depth_mm": None if runtime_reference is None else runtime_reference.get("depth_mm"),
        "runtime_angle_deg": None if runtime_reference is None else runtime_reference.get("angle_deg"),
        "delta_x_px": delta.get("delta_x_px"),
        "delta_y_px": delta.get("delta_y_px"),
        "delta_angle_deg": delta.get("delta_angle_deg"),
        "delta_depth_mm": delta.get("delta_depth_mm"),
        "analysis_image_path": str(analysis_image_path),
    }


def save_named_rgb_images(output_case_dir: Path, images: Dict[str, np.ndarray], file_map: Dict[str, str]) -> Dict[str, str]:
    saved_paths: dict[str, str] = {}
    for key, file_name in file_map.items():
        image = images.get(key)
        if image is None:
            continue
        path = output_case_dir / file_name
        save_rgb_png(path, image)
        saved_paths[key] = str(path)
    return saved_paths


def analyze_primary_capture(
    *,
    case_id: str,
    case_dir: Path,
    metadata: Dict[str, Any],
    recognizer: Any,
    tolerances: Tolerances,
    output_case_dir: Path,
) -> Optional[Dict[str, Any]]:
    arrays = load_capture_arrays(case_dir, PRIMARY_CAPTURE)
    if arrays is None:
        return None
    rgb, depth = arrays
    inference_result = run_primary_review_inference(recognizer, rgb, depth)
    candidate = inference_result["selected_candidate"]
    offline_candidates = list(inference_result["ranked_candidates"])
    selected_rank_index = inference_result["selected_rank_index"]

    runtime_reference = extract_primary_runtime_reference(metadata)
    status, delta = compute_primary_status(candidate, runtime_reference, tolerances)
    primary_analysis_image = build_primary_analysis_image(
        rgb,
        depth,
        candidate,
        offline_candidates,
        selected_rank_index,
        runtime_reference,
        case_id,
        status,
        delta,
    )

    depth_report = run_depth_review_inference(recognizer, rgb, depth)
    depth_selected = depth_report["selected_candidate"]
    depth_candidates = list(depth_report["ranked_candidates"])
    depth_runtime_proxy_metrics = compute_depth_runtime_proxy_metrics(depth_selected, runtime_reference, tolerances)
    depth_manual_gt_metrics = compute_depth_manual_ground_truth_metrics(depth_selected, metadata, recognizer, tolerances)
    depth_status = depth_manual_gt_metrics["status"]
    if depth_status == "gt_missing":
        depth_status = depth_runtime_proxy_metrics["status"]

    depth_debug_images = dict(depth_report["debug_images"])
    depth_analysis_image = build_depth_analysis_image(
        depth_debug_images.get("analysis_panel", depth_report["depth_debug"]),
        case_id=case_id,
        board_depth_mm=depth_report["board_depth_mm"],
        depth_status=depth_status,
        manual_gt_metrics=depth_manual_gt_metrics,
        runtime_proxy_metrics=depth_runtime_proxy_metrics,
        selected_candidate=depth_selected,
    )
    analysis_image = combine_primary_and_depth_analysis(primary_analysis_image, depth_analysis_image)
    image_path = output_case_dir / "primary_pick_analysis.png"
    save_rgb_png(image_path, analysis_image)
    depth_debug_images["depth_analysis"] = depth_analysis_image
    depth_image_paths = save_named_rgb_images(
        output_case_dir,
        depth_debug_images,
        {
            "height_map": "primary_pick_depth_height_map.png",
            "foreground_mask": "primary_pick_depth_foreground_mask.png",
            "regions": "primary_pick_depth_regions.png",
            "rectangles": "primary_pick_depth_rectangles.png",
            "depth_analysis": "primary_pick_depth_analysis.png",
        },
    )
    depth_candidates_path = output_case_dir / "primary_pick_depth_candidates.json"
    depth_candidates_path.write_text(
        json.dumps(
            to_serializable(
                {
                    "board_depth_mm": depth_report["board_depth_mm"],
                    "workspace_roi": depth_report["workspace_roi"],
                    "selected_candidate": sanitize_depth_candidate(depth_selected),
                    "candidates": sanitize_depth_candidates(depth_candidates, depth_selected),
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    sanitized_offline = sanitize_primary_candidate(candidate)
    sanitized_candidates = sanitize_primary_candidates(offline_candidates, candidate)
    row = build_stage_row(
        case_id=case_id,
        capture_stage=PRIMARY_CAPTURE,
        status=status,
        offline_result=sanitized_offline,
        offline_candidate_count=len(offline_candidates),
        offline_selected_rank=(None if selected_rank_index is None else selected_rank_index + 1),
        runtime_reference=runtime_reference,
        delta=delta,
        analysis_image_path=image_path,
    )
    row.update(
        {
            "depth_selected_exists": depth_selected is not None,
            "depth_candidate_count": len(depth_candidates),
            "depth_selected_geometry_score": None if depth_selected is None else depth_selected.get("geometry_score"),
            "depth_selected_robot_x_mm": depth_manual_gt_metrics.get("robot_x_mm"),
            "depth_selected_robot_y_mm": depth_manual_gt_metrics.get("robot_y_mm"),
            "depth_delta_robot_x_mm": depth_manual_gt_metrics.get("delta_robot_x_mm"),
            "depth_delta_robot_y_mm": depth_manual_gt_metrics.get("delta_robot_y_mm"),
            "depth_delta_pick_z_mm": depth_manual_gt_metrics.get("delta_pick_z_mm"),
            "depth_delta_j6_delta_deg": depth_manual_gt_metrics.get("delta_j6_delta_deg"),
            "depth_status": depth_status,
        }
    )
    return {
        "row": row,
        "image_path": image_path,
        "extra_image_paths": [
            Path(depth_image_paths["depth_analysis"])
        ] if "depth_analysis" in depth_image_paths else [],
        "stage_report": {
            "capture_stage": PRIMARY_CAPTURE,
            "offline_detected": candidate is not None,
            "runtime_reference_detected": runtime_reference is not None,
            "status": status,
            "offline_candidate_count": len(offline_candidates),
            "offline_selected_rank": (None if selected_rank_index is None else selected_rank_index + 1),
            "offline_result": sanitized_offline,
            "offline_candidates": sanitized_candidates,
            "runtime_reference": runtime_reference,
            "delta": delta,
            "analysis_image_path": str(image_path),
            "depth_geometry": {
                "board_depth_mm": depth_report["board_depth_mm"],
                "workspace_roi": depth_report["workspace_roi"],
                "selected_candidate": sanitize_depth_candidate(depth_selected),
                "candidates": sanitize_depth_candidates(depth_candidates, depth_selected),
                "manual_gt_metrics": depth_manual_gt_metrics,
                "runtime_proxy_metrics": depth_runtime_proxy_metrics,
                "analysis_image_path": depth_image_paths.get("depth_analysis"),
                "debug_image_paths": depth_image_paths,
                "depth_candidates_json_path": str(depth_candidates_path),
                "status": depth_status,
            },
        },
    }


def analyze_secondary_capture(
    *,
    case_id: str,
    case_dir: Path,
    metadata: Dict[str, Any],
    recognizer: Any,
    tolerances: Tolerances,
    output_case_dir: Path,
) -> Optional[Dict[str, Any]]:
    arrays = load_capture_arrays(case_dir, SECONDARY_CAPTURE)
    if arrays is None:
        return None
    rgb, depth = arrays
    display_rgb, alignment_data = recognizer.blinx_brickandporcelain_image_rec(rgb)
    offline_result = sanitize_secondary_result(alignment_data)
    runtime_reference = extract_secondary_runtime_reference(metadata)
    status, delta = compute_secondary_status(offline_result, runtime_reference, tolerances)
    secondary_analysis_image = build_secondary_analysis_image(
        display_rgb,
        depth,
        offline_result,
        runtime_reference,
        case_id,
        status,
        delta,
    )

    depth_report = run_secondary_depth_review_inference(recognizer, rgb, depth)
    depth_selected = depth_report["selected_candidate"]
    depth_candidates = list(depth_report["ranked_candidates"])
    depth_runtime_proxy_metrics = compute_depth_runtime_proxy_metrics(depth_selected, runtime_reference, tolerances)
    depth_status = depth_runtime_proxy_metrics["status"]

    depth_debug_images = dict(depth_report["debug_images"])
    depth_analysis_image = build_secondary_depth_analysis_image(
        depth_debug_images.get("analysis_panel", depth_report["depth_debug"]),
        case_id=case_id,
        board_depth_mm=depth_report["board_depth_mm"],
        depth_status=depth_status,
        runtime_proxy_metrics=depth_runtime_proxy_metrics,
        selected_candidate=depth_selected,
    )
    analysis_image = combine_primary_and_depth_analysis(secondary_analysis_image, depth_analysis_image)
    image_path = output_case_dir / "secondary_alignment_analysis.png"
    save_rgb_png(image_path, analysis_image)
    depth_debug_images["depth_analysis"] = depth_analysis_image
    depth_image_paths = save_named_rgb_images(
        output_case_dir,
        depth_debug_images,
        {
            "height_map": "secondary_alignment_depth_height_map.png",
            "foreground_mask": "secondary_alignment_depth_foreground_mask.png",
            "regions": "secondary_alignment_depth_regions.png",
            "rectangles": "secondary_alignment_depth_rectangles.png",
            "depth_analysis": "secondary_alignment_depth_analysis.png",
        },
    )
    depth_candidates_path = output_case_dir / "secondary_alignment_depth_candidates.json"
    depth_candidates_path.write_text(
        json.dumps(
            to_serializable(
                {
                    "board_depth_mm": depth_report["board_depth_mm"],
                    "workspace_roi": depth_report["workspace_roi"],
                    "selected_candidate": sanitize_depth_candidate(depth_selected),
                    "candidates": sanitize_depth_candidates(depth_candidates, depth_selected),
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    row = build_stage_row(
        case_id=case_id,
        capture_stage=SECONDARY_CAPTURE,
        status=status,
        offline_result=offline_result,
        offline_candidate_count=(1 if offline_result is not None else 0),
        offline_selected_rank=(1 if offline_result is not None else None),
        runtime_reference=runtime_reference,
        delta=delta,
        analysis_image_path=image_path,
    )
    row.update(
        {
            "depth_selected_exists": depth_selected is not None,
            "depth_candidate_count": len(depth_candidates),
            "depth_selected_geometry_score": None if depth_selected is None else depth_selected.get("geometry_score"),
            "depth_selected_robot_x_mm": None,
            "depth_selected_robot_y_mm": None,
            "depth_delta_robot_x_mm": None,
            "depth_delta_robot_y_mm": None,
            "depth_delta_pick_z_mm": None,
            "depth_delta_j6_delta_deg": None,
            "depth_status": depth_status,
        }
    )
    return {
        "row": row,
        "image_path": image_path,
        "extra_image_paths": [
            Path(depth_image_paths["depth_analysis"])
        ] if "depth_analysis" in depth_image_paths else [],
        "stage_report": {
            "capture_stage": SECONDARY_CAPTURE,
            "offline_detected": offline_result is not None,
            "runtime_reference_detected": runtime_reference is not None,
            "status": status,
            "offline_candidate_count": (1 if offline_result is not None else 0),
            "offline_selected_rank": (1 if offline_result is not None else None),
            "offline_result": offline_result,
            "runtime_reference": runtime_reference,
            "delta": delta,
            "analysis_image_path": str(image_path),
            "depth_geometry": {
                "board_depth_mm": depth_report["board_depth_mm"],
                "workspace_roi": depth_report["workspace_roi"],
                "selected_candidate": sanitize_depth_candidate(depth_selected),
                "candidates": sanitize_depth_candidates(depth_candidates, depth_selected),
                "runtime_proxy_metrics": depth_runtime_proxy_metrics,
                "analysis_image_path": depth_image_paths.get("depth_analysis"),
                "debug_image_paths": depth_image_paths,
                "depth_candidates_json_path": str(depth_candidates_path),
                "status": depth_status,
            },
        },
    }


def analyze_case(
    case_dir: Path,
    recognizer: Any,
    capture_scope: Sequence[str],
    tolerances: Tolerances,
    output_root: Path,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Path]]:
    case_id = case_dir.name
    metadata = load_metadata(case_dir)
    output_case_dir = output_root / case_id
    output_case_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    image_paths: List[Path] = []
    stage_reports: Dict[str, Any] = {}

    if PRIMARY_CAPTURE in capture_scope:
        primary_result = analyze_primary_capture(
            case_id=case_id,
            case_dir=case_dir,
            metadata=metadata,
            recognizer=recognizer,
            tolerances=tolerances,
            output_case_dir=output_case_dir,
        )
        if primary_result is not None:
            rows.append(primary_result["row"])
            image_paths.append(primary_result["image_path"])
            stage_reports[PRIMARY_CAPTURE] = primary_result["stage_report"]

    if SECONDARY_CAPTURE in capture_scope:
        secondary_result = analyze_secondary_capture(
            case_id=case_id,
            case_dir=case_dir,
            metadata=metadata,
            recognizer=recognizer,
            tolerances=tolerances,
            output_case_dir=output_case_dir,
        )
        if secondary_result is not None:
            rows.append(secondary_result["row"])
            image_paths.append(secondary_result["image_path"])
            stage_reports[SECONDARY_CAPTURE] = secondary_result["stage_report"]

    case_report = {
        "case_id": case_id,
        "session_name": metadata.get("session_name"),
        "source_case_dir": str(case_dir),
        "stages": stage_reports,
    }
    (output_case_dir / "analysis.json").write_text(
        json.dumps(to_serializable(case_report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return case_report, rows, image_paths


def write_summary_files(output_root: Path, summary_rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "case_id",
        "capture_stage",
        "offline_detected",
        "runtime_reference_detected",
        "status",
        "offline_candidate_count",
        "offline_selected_rank",
        "offline_pixel_x",
        "offline_pixel_y",
        "offline_depth_mm",
        "offline_angle_deg",
        "offline_score",
        "runtime_pixel_x",
        "runtime_pixel_y",
        "runtime_depth_mm",
        "runtime_angle_deg",
        "delta_x_px",
        "delta_y_px",
        "delta_angle_deg",
        "delta_depth_mm",
        "analysis_image_path",
        "depth_selected_exists",
        "depth_candidate_count",
        "depth_selected_geometry_score",
        "depth_selected_robot_x_mm",
        "depth_selected_robot_y_mm",
        "depth_delta_robot_x_mm",
        "depth_delta_robot_y_mm",
        "depth_delta_pick_z_mm",
        "depth_delta_j6_delta_deg",
        "depth_status",
    ]

    summary_csv_path = output_root / "summary.csv"
    with summary_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    review_rows = [row for row in summary_rows if row["status"] != "runtime_consistent"]
    review_csv_path = output_root / "review_queue.csv"
    with review_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in review_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    status_counts: Dict[str, int] = {}
    for row in summary_rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    summary_json = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "summary_count": len(summary_rows),
        "review_count": len(review_rows),
        "status_counts": status_counts,
        "rows": list(summary_rows),
    }
    (output_root / "summary.json").write_text(
        json.dumps(to_serializable(summary_json), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def browse_images(image_paths: Sequence[Path]) -> None:
    if not image_paths:
        return

    window_name = "offline_brick_recognition_review"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    screen_size = get_screen_size()
    index = 0
    try:
        while True:
            image = load_bgr_image(image_paths[index])
            display_image, scale = fit_image_to_screen(image, screen_size=screen_size)
            cv2.resizeWindow(window_name, display_image.shape[1], display_image.shape[0])
            cv2.imshow(window_name, display_image)
            print(
                f"[{index + 1}/{len(image_paths)}] {image_paths[index].name}  "
                f"original={image.shape[1]}x{image.shape[0]} display={display_image.shape[1]}x{display_image.shape[0]} "
                f"scale={scale:.3f}  keys: n=next p=prev q=quit"
            )
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                index = (index - 1) % len(image_paths)
            else:
                index = (index + 1) % len(image_paths)
    finally:
        cv2.destroyAllWindows()


def run_review(args: argparse.Namespace) -> Dict[str, Any]:
    session_dir = resolve_session_dir(args.session_dir)
    output_root = resolve_output_dir(session_dir, args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    recognizer = create_recognizer(args)
    tolerances = Tolerances(
        px=args.px_tol,
        angle_deg=args.angle_tol_deg,
        depth_mm=args.depth_tol_mm,
        robot_xy_mm=args.robot_xy_tol_mm,
    )

    capture_scope = capture_scopes(args.capture_scope)
    summary_rows: List[Dict[str, Any]] = []
    case_reports: List[Dict[str, Any]] = []
    image_paths: List[Path] = []

    for case_dir in list_case_dirs(session_dir, args.case_id):
        case_report, rows, case_image_paths = analyze_case(
            case_dir=case_dir,
            recognizer=recognizer,
            capture_scope=capture_scope,
            tolerances=tolerances,
            output_root=output_root,
        )
        case_reports.append(case_report)
        summary_rows.extend(rows)
        image_paths.extend(case_image_paths)

    write_summary_files(output_root, summary_rows)
    return {
        "session_dir": session_dir,
        "output_root": output_root,
        "case_reports": case_reports,
        "summary_rows": summary_rows,
        "image_paths": image_paths,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = run_review(args)
    except Exception as exc:
        print(f"offline_brick_recognition_review failed: {exc}")
        return 1

    print(f"Session: {result['session_dir']}")
    print(f"Output: {result['output_root']}")
    print(f"Analyzed rows: {len(result['summary_rows'])}")

    status_counts: Dict[str, int] = {}
    for row in result["summary_rows"]:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    if status_counts:
        print("Status counts:")
        for key in sorted(status_counts):
            print(f"  {key}: {status_counts[key]}")

    if not args.no_window:
        try:
            browse_images(result["image_paths"])
        except cv2.error as exc:
            print(f"OpenCV window browsing unavailable: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
