from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def build_display_rgb(rgb_image: np.ndarray) -> np.ndarray:
    rgb_image = np.asarray(rgb_image)
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("RGB image must have shape (height, width, 3)")
    return np.ascontiguousarray(rgb_image.copy())


def build_depth_preview(depth_map: np.ndarray) -> np.ndarray:
    depth_map = np.asarray(depth_map)
    if depth_map.ndim != 2:
        raise ValueError("Depth map must have shape (height, width)")
    depth_view = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
    depth_8bit = cv2.normalize(depth_view, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    depth_bgr = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
    return cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)


def save_rgb_png(path: Path, rgb_image: np.ndarray) -> None:
    bgr_image = cv2.cvtColor(np.ascontiguousarray(rgb_image), cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".png", bgr_image)
    if not success:
        raise IOError(f"Failed to encode image: {path}")
    try:
        encoded.tofile(str(path))
    except OSError as exc:
        raise IOError(f"Failed to save image: {path}") from exc


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


def build_display_panel(
    display_rgb: np.ndarray,
    depth_preview: np.ndarray,
    *,
    case_id: str,
    capture_name: str,
    captured_at: str,
) -> np.ndarray:
    header_height = 96
    target_height = max(display_rgb.shape[0], depth_preview.shape[0])
    left = _pad_to_height(display_rgb, target_height)
    right = _pad_to_height(depth_preview, target_height)

    separator = np.full((target_height, 16, 3), 255, dtype=np.uint8)
    content = np.hstack((left, separator, right))
    panel = np.full((header_height + content.shape[0], content.shape[1], 3), 255, dtype=np.uint8)
    panel[header_height:, :, :] = content

    text_lines = [
        f"Case: {case_id}",
        f"Capture: {capture_name}",
        f"Captured At: {captured_at}",
    ]
    for index, line in enumerate(text_lines):
        origin = (16, 28 + index * 24)
        cv2.putText(panel, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    return panel


class BrickProcessRecorder:
    def __init__(
        self,
        *,
        base_dir: Path | str = Path("pic") / "cam1" / "brick_process_records",
        session_name: Optional[str] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.session_name = session_name or datetime.now().strftime("brick_%Y%m%d_%H%M%S")
        self.session_dir = (self.base_dir / self.session_name).resolve()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.case_counter = 0
        self.current_case_id: Optional[str] = None
        self.current_case_dir: Optional[Path] = None
        self.current_case_metadata: Optional[Dict[str, Any]] = None

    def _next_case_id(self) -> str:
        self.case_counter += 1
        return f"case_{self.case_counter:04d}"

    def _flush_metadata(self) -> None:
        if self.current_case_dir is None or self.current_case_metadata is None:
            return
        metadata_path = self.current_case_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(_to_serializable(self.current_case_metadata), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def start_case(
        self,
        *,
        process_num: int,
        process_node: str,
        raw_rgb: np.ndarray,
        depth_map: np.ndarray,
        display_rgb: np.ndarray,
        public_snapshot: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        if self.current_case_id is not None:
            self.finalize_case(
                status="interrupted_by_new_case",
                process_node=process_node,
                public_snapshot=public_snapshot,
                extra={"reason": "A new case started before the previous case was finalized."},
            )

        case_id = self._next_case_id()
        case_dir = self.session_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=False)

        self.current_case_id = case_id
        self.current_case_dir = case_dir
        self.current_case_metadata = {
            "case_id": case_id,
            "session_name": self.session_name,
            "case_type": "brick_process_cycle",
            "created_at": now_iso(),
            "status": "in_progress",
            "brick_process_num_at_start": int(process_num),
            "initial_process_node": process_node,
            "captures": {},
            "events": [],
            "manual_ground_truth_tcp_pose": None,
            "manual_ground_truth_joint_pos": None,
            "manual_ground_truth_recorded_at": None,
        }
        self.record_capture(
            capture_name="primary_pick",
            raw_rgb=raw_rgb,
            depth_map=depth_map,
            display_rgb=display_rgb,
            process_node=process_node,
            public_snapshot=public_snapshot,
            extra=extra,
        )
        self.record_event(
            event_name="primary_pick_recorded",
            process_node=process_node,
            public_snapshot=public_snapshot,
            command_target=(extra or {}).get("command_target"),
            extra=extra,
        )
        return case_id

    def record_capture(
        self,
        *,
        capture_name: str,
        raw_rgb: np.ndarray,
        depth_map: np.ndarray,
        display_rgb: Optional[np.ndarray],
        process_node: str,
        public_snapshot: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.current_case_id is None or self.current_case_dir is None or self.current_case_metadata is None:
            return

        captured_at = now_iso()
        raw_rgb = np.ascontiguousarray(np.asarray(raw_rgb))
        depth_map = np.ascontiguousarray(np.asarray(depth_map))
        display_rgb = build_display_rgb(raw_rgb if display_rgb is None else display_rgb)
        depth_preview = build_depth_preview(depth_map)
        display_panel = build_display_panel(
            display_rgb,
            depth_preview,
            case_id=self.current_case_id,
            capture_name=capture_name,
            captured_at=captured_at,
        )

        prefix = capture_name
        file_names = {
            "rgb_npy": f"{prefix}_rgb.npy",
            "depth_npy": f"{prefix}_depth.npy",
            "rgb_raw_png": f"{prefix}_rgb_raw.png",
            "display_rgb_png": f"{prefix}_display_rgb.png",
            "depth_preview_png": f"{prefix}_depth_preview.png",
            "display_panel_png": f"{prefix}_display_panel.png",
        }

        np.save(self.current_case_dir / file_names["rgb_npy"], raw_rgb)
        np.save(self.current_case_dir / file_names["depth_npy"], depth_map)
        save_rgb_png(self.current_case_dir / file_names["rgb_raw_png"], raw_rgb)
        save_rgb_png(self.current_case_dir / file_names["display_rgb_png"], display_rgb)
        save_rgb_png(self.current_case_dir / file_names["depth_preview_png"], depth_preview)
        save_rgb_png(self.current_case_dir / file_names["display_panel_png"], display_panel)

        self.current_case_metadata["captures"][capture_name] = {
            "captured_at": captured_at,
            "process_node": process_node,
            "rgb_shape": list(raw_rgb.shape),
            "rgb_dtype": str(raw_rgb.dtype),
            "depth_shape": list(depth_map.shape),
            "depth_dtype": str(depth_map.dtype),
            "files": file_names,
            "public_snapshot": _to_serializable(public_snapshot or {}),
            "extra": _to_serializable(extra or {}),
        }
        self._flush_metadata()

    def record_event(
        self,
        *,
        event_name: str,
        process_node: str,
        public_snapshot: Optional[Dict[str, Any]] = None,
        command_target: Optional[Sequence[float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.current_case_metadata is None:
            return

        event = {
            "timestamp": now_iso(),
            "event_name": event_name,
            "process_node": process_node,
            "public_snapshot": _to_serializable(public_snapshot or {}),
            "command_target": _to_serializable(command_target),
            "extra": _to_serializable(extra or {}),
        }
        self.current_case_metadata["events"].append(event)
        self._flush_metadata()

    def finalize_case(
        self,
        *,
        status: str,
        process_node: str,
        public_snapshot: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.current_case_metadata is None:
            return

        self.current_case_metadata["status"] = status
        self.current_case_metadata["finalized_at"] = now_iso()
        self.current_case_metadata["final_process_node"] = process_node
        self.current_case_metadata["final_public_snapshot"] = _to_serializable(public_snapshot or {})
        self.current_case_metadata["final_extra"] = _to_serializable(extra or {})
        self._flush_metadata()
        self.current_case_id = None
        self.current_case_dir = None
        self.current_case_metadata = None
