import math
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from ImageProcessing.BLX_Image_Camera import Blinx_Image_Camera
from ImageProcessing.depth_brick_geometry import (
    build_depth_candidate,
    build_height_map,
    extract_depth_foreground,
    fit_region_rectangle,
    preprocess_depth_roi,
    rank_depth_candidates,
    render_depth_debug_panel,
    split_connected_regions,
)
from ImageProcessing.yolov8_onnx import YOLOv8
from ImageProcessing.yolov8_onnx_seg import YOLOv8Seg
from ImageProcessing.yolov8_onnx2 import YOLOv82


def normalize_angle_90(angle_deg: float) -> float:
    return ((float(angle_deg) + 90.0) % 180.0) - 90.0


def resolve_shortest_rotation_delta(previous_angle: Optional[float], angle_deg: float) -> float:
    if previous_angle is None:
        return float(angle_deg)

    candidates = [float(angle_deg), float(angle_deg) + 180.0, float(angle_deg) - 180.0]
    return min(candidates, key=lambda candidate: abs(candidate - float(previous_angle)))


def calculate_rectangle_angle(corners) -> float:
    if len(corners) != 4:
        raise ValueError("four corner points are required")

    corners = np.asarray(corners, dtype=np.float32)
    edges = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        edge_length = np.sqrt(np.sum((p1 - p2) ** 2))
        edges.append((edge_length, (p1, p2)))

    edges.sort(key=lambda item: item[0], reverse=True)
    longest_edge = edges[0][1]
    p1, p2 = longest_edge
    vector = p2 - p1
    angle_rad = np.arctan2(vector[1], vector[0])
    angle_deg = np.degrees(angle_rad)
    return normalize_angle_90(angle_deg)


def prepare_mask(mask: np.ndarray, erode_kernel: int, erode_iterations: int) -> np.ndarray:
    mask_bool = np.asarray(mask > 0, dtype=bool)
    if not np.any(mask_bool):
        return mask_bool

    erode_kernel = max(int(erode_kernel), 1)
    erode_iterations = max(int(erode_iterations), 0)
    if erode_kernel <= 1 or erode_iterations == 0:
        return mask_bool

    kernel = np.ones((erode_kernel, erode_kernel), dtype=np.uint8)
    eroded = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=erode_iterations)
    return eroded.astype(bool)


def _mask_centroid(mask: np.ndarray) -> Optional[tuple]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return int(np.round(xs.mean())), int(np.round(ys.mean()))


def _build_depth_stat_result(
    original_mask: np.ndarray,
    working_mask: np.ndarray,
    depth_map: np.ndarray,
    cfg: Dict[str, float],
) -> Dict[str, object]:
    total_foreground = int(np.count_nonzero(original_mask))
    ys, xs = np.where(working_mask > 0)
    if xs.size == 0 or total_foreground == 0:
        centroid = _mask_centroid(original_mask)
        return {
            "depth_mm": None,
            "pixel_x": centroid[0] if centroid else None,
            "pixel_y": centroid[1] if centroid else None,
            "valid_depth_count": 0,
            "valid_depth_ratio": 0.0,
            "inlier_mask": np.zeros_like(original_mask, dtype=bool),
            "rejected_mask": original_mask.astype(bool),
            "is_valid": False,
        }

    depth_values = depth_map[ys, xs].astype(np.float32)
    finite_mask = np.isfinite(depth_values) & (depth_values > float(cfg["depth_min_valid_mm"]))
    valid_depth_values = depth_values[finite_mask]
    valid_xs = xs[finite_mask]
    valid_ys = ys[finite_mask]

    inlier_mask = np.zeros_like(original_mask, dtype=bool)
    rejected_mask = original_mask.astype(bool)
    if valid_depth_values.size == 0:
        centroid = _mask_centroid(original_mask)
        return {
            "depth_mm": None,
            "pixel_x": centroid[0] if centroid else None,
            "pixel_y": centroid[1] if centroid else None,
            "valid_depth_count": 0,
            "valid_depth_ratio": 0.0,
            "inlier_mask": inlier_mask,
            "rejected_mask": rejected_mask,
            "is_valid": False,
        }

    trim_percent = max(0.0, min(float(cfg["depth_trim_percent"]), 49.0))
    if trim_percent > 0.0 and valid_depth_values.size > 2:
        low = np.percentile(valid_depth_values, trim_percent)
        high = np.percentile(valid_depth_values, 100.0 - trim_percent)
        trimmed_mask = (valid_depth_values >= low) & (valid_depth_values <= high)
    else:
        trimmed_mask = np.ones_like(valid_depth_values, dtype=bool)

    inlier_depth_values = valid_depth_values[trimmed_mask]
    inlier_xs = valid_xs[trimmed_mask]
    inlier_ys = valid_ys[trimmed_mask]
    if inlier_depth_values.size > 0:
        inlier_mask[inlier_ys, inlier_xs] = True
        rejected_mask[inlier_ys, inlier_xs] = False

    valid_depth_count = int(inlier_depth_values.size)
    valid_depth_ratio = valid_depth_count / float(max(total_foreground, 1))
    centroid = _mask_centroid(original_mask)
    if valid_depth_count > 0:
        pixel_x = int(np.round(inlier_xs.mean()))
        pixel_y = int(np.round(inlier_ys.mean()))
        depth_mm = float(np.median(inlier_depth_values))
    else:
        pixel_x = centroid[0] if centroid else None
        pixel_y = centroid[1] if centroid else None
        depth_mm = None

    is_valid = (
        depth_mm is not None
        and valid_depth_count >= int(cfg["depth_min_valid_pixels"])
        and valid_depth_ratio >= float(cfg["depth_min_valid_ratio"])
    )
    return {
        "depth_mm": depth_mm,
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "valid_depth_count": valid_depth_count,
        "valid_depth_ratio": valid_depth_ratio,
        "inlier_mask": inlier_mask,
        "rejected_mask": rejected_mask,
        "is_valid": is_valid,
    }


def extract_depth_stats(mask: np.ndarray, depth_map: np.ndarray, cfg: Dict[str, float]) -> Dict[str, object]:
    original_mask = np.asarray(mask > 0, dtype=bool)
    if depth_map is None:
        centroid = _mask_centroid(original_mask)
        return {
            "depth_mm": None,
            "pixel_x": centroid[0] if centroid else None,
            "pixel_y": centroid[1] if centroid else None,
            "valid_depth_count": 0,
            "valid_depth_ratio": 0.0,
            "inlier_mask": np.zeros_like(original_mask, dtype=bool),
            "rejected_mask": original_mask.astype(bool),
            "is_valid": False,
        }

    eroded_mask = prepare_mask(
        original_mask,
        cfg["depth_erode_kernel"],
        cfg["depth_erode_iterations"],
    )
    stats = _build_depth_stat_result(original_mask, eroded_mask, depth_map, cfg)
    if stats["valid_depth_count"] < int(cfg["depth_min_valid_pixels"]) and np.any(eroded_mask != original_mask):
        stats = _build_depth_stat_result(original_mask, original_mask, depth_map, cfg)
    return stats


def _compute_min_area_rect_angle(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if xs.size < 2:
        return 0.0
    points = np.column_stack((xs, ys)).astype(np.float32)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    return calculate_rectangle_angle(box)


def compute_pca_angle(mask: np.ndarray, cfg: Dict[str, float]) -> Dict[str, object]:
    ys, xs = np.where(mask > 0)
    if xs.size < 2:
        fallback_angle = _compute_min_area_rect_angle(mask)
        return {
            "angle_deg": fallback_angle,
            "raw_pca_angle_deg": fallback_angle,
            "axis_ratio": 0.0,
            "angle_fallback": True,
        }

    points = np.column_stack((xs, ys)).astype(np.float32)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean=np.empty((0)))
    del mean
    raw_angle_deg = normalize_angle_90(np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])))

    if len(eigenvalues) >= 2:
        axis_ratio = float(eigenvalues[0, 0] / max(eigenvalues[1, 0], 1e-6))
    else:
        axis_ratio = float("inf")

    if not np.isfinite(axis_ratio) or axis_ratio < float(cfg["pca_min_axis_ratio"]):
        fallback_angle = _compute_min_area_rect_angle(mask)
        return {
            "angle_deg": fallback_angle,
            "raw_pca_angle_deg": raw_angle_deg,
            "axis_ratio": axis_ratio,
            "angle_fallback": True,
        }

    return {
        "angle_deg": raw_angle_deg,
        "raw_pca_angle_deg": raw_angle_deg,
        "axis_ratio": axis_ratio,
        "angle_fallback": False,
    }


def rank_candidates(candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        candidates,
        key=lambda item: (
            not item["is_valid"],
            item["depth_mm"] if item["depth_mm"] is not None else float("inf"),
            -float(item["score"]),
        ),
    )


def _candidate_float(candidate: Optional[Dict[str, object]], key: str, default: float = 0.0) -> float:
    if not isinstance(candidate, dict):
        return float(default)
    value = candidate.get(key, default)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _compute_bbox_iou(box_a: Optional[List[float]], box_b: Optional[List[float]]) -> float:
    if box_a is None or box_b is None or len(box_a) < 4 or len(box_b) < 4:
        return 0.0

    ax1, ay1, ax2, ay2 = [float(value) for value in box_a[:4]]
    bx1, by1, bx2, by2 = [float(value) for value in box_b[:4]]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(inter_x2 - inter_x1, 0.0)
    inter_h = max(inter_y2 - inter_y1, 0.0)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def _compute_mask_iou(mask_a: Optional[np.ndarray], mask_b: Optional[np.ndarray]) -> float:
    if mask_a is None or mask_b is None:
        return 0.0
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    if mask_a.shape != mask_b.shape:
        return 0.0
    union = np.logical_or(mask_a, mask_b)
    union_count = int(np.count_nonzero(union))
    if union_count == 0:
        return 0.0
    intersection = int(np.count_nonzero(np.logical_and(mask_a, mask_b)))
    return float(intersection / union_count)


def evaluate_rgb_candidate_quality(candidate: Optional[Dict[str, object]], cfg: Dict[str, float]) -> Dict[str, object]:
    if candidate is None:
        return {"is_low_quality": True, "reasons": ["rgb_missing"]}

    reasons: List[str] = []
    if _candidate_float(candidate, "score") < float(cfg["primary_pick_rgb_low_score_thresh"]):
        reasons.append("low_score")
    if _candidate_float(candidate, "valid_depth_ratio") < float(cfg["primary_pick_rgb_low_valid_ratio_thresh"]):
        reasons.append("low_valid_depth_ratio")
    if bool(candidate.get("angle_fallback")):
        reasons.append("angle_fallback")
    if not bool(candidate.get("is_valid", True)):
        reasons.append("invalid_depth_support")
    return {
        "is_low_quality": bool(reasons),
        "reasons": reasons,
    }


def evaluate_depth_fallback_candidate(candidate: Optional[Dict[str, object]], cfg: Dict[str, float]) -> Dict[str, object]:
    if candidate is None:
        return {"is_ready": False, "reasons": ["depth_missing"]}

    reasons: List[str] = []
    if not bool(candidate.get("is_valid")):
        reasons.append("invalid_candidate")
    if bool(candidate.get("touch_border")):
        reasons.append("touch_border")
    if _candidate_float(candidate, "geometry_score") < float(cfg["primary_pick_depth_fallback_geom_thresh"]):
        reasons.append("low_geometry_score")
    return {
        "is_ready": not reasons,
        "reasons": reasons,
    }


def match_depth_candidate_to_rgb(
    rgb_candidate: Optional[Dict[str, object]],
    depth_candidates: List[Dict[str, object]],
    cfg: Dict[str, float],
) -> Dict[str, object]:
    if rgb_candidate is None:
        return {
            "match_found": False,
            "matched_candidate": None,
            "metrics": None,
        }

    best_candidate = None
    best_metrics = None
    best_sort_key = None
    for depth_candidate in depth_candidates:
        if not bool(depth_candidate.get("is_valid")):
            continue

        center_distance_px = math.hypot(
            _candidate_float(depth_candidate, "pixel_x") - _candidate_float(rgb_candidate, "pixel_x"),
            _candidate_float(depth_candidate, "pixel_y") - _candidate_float(rgb_candidate, "pixel_y"),
        )
        angle_delta_deg = abs(
            normalize_angle_90(
                _candidate_float(depth_candidate, "angle_deg") - _candidate_float(rgb_candidate, "angle_deg")
            )
        )
        depth_delta_mm = abs(
            _candidate_float(depth_candidate, "depth_mm") - _candidate_float(rgb_candidate, "depth_mm")
        )
        bbox_iou = _compute_bbox_iou(
            rgb_candidate.get("bbox"),
            depth_candidate.get("bbox"),
        )
        mask_iou = _compute_mask_iou(
            rgb_candidate.get("mask"),
            depth_candidate.get("mask"),
        )

        metrics = {
            "center_distance_px": float(center_distance_px),
            "angle_delta_deg": float(angle_delta_deg),
            "depth_delta_mm": float(depth_delta_mm),
            "bbox_iou": float(bbox_iou),
            "mask_iou": float(mask_iou),
        }
        passes = (
            metrics["center_distance_px"] <= float(cfg["primary_pick_rgb_depth_center_thresh_px"])
            and metrics["angle_delta_deg"] <= float(cfg["primary_pick_rgb_depth_angle_thresh_deg"])
            and metrics["depth_delta_mm"] <= float(cfg["primary_pick_rgb_depth_mm_thresh"])
            and metrics["mask_iou"] >= float(cfg["primary_pick_rgb_depth_iou_thresh"])
        )
        if not passes:
            continue

        sort_key = (
            metrics["mask_iou"],
            metrics["bbox_iou"],
            -metrics["center_distance_px"],
            -metrics["angle_delta_deg"],
            -metrics["depth_delta_mm"],
            _candidate_float(depth_candidate, "geometry_score"),
        )
        if best_sort_key is None or sort_key > best_sort_key:
            best_sort_key = sort_key
            best_candidate = depth_candidate
            best_metrics = metrics

    return {
        "match_found": best_candidate is not None,
        "matched_candidate": best_candidate,
        "metrics": best_metrics,
    }


def select_primary_pick_candidate(
    rgb_report: Dict[str, object],
    depth_report: Dict[str, object],
    cfg: Dict[str, float],
) -> Dict[str, object]:
    rgb_selected = rgb_report.get("selected_candidate")
    depth_selected = depth_report.get("selected_candidate")
    depth_candidates = list(depth_report.get("ranked_candidates") or [])

    rgb_quality = evaluate_rgb_candidate_quality(rgb_selected, cfg)
    depth_fallback = evaluate_depth_fallback_candidate(depth_selected, cfg)
    match_report = match_depth_candidate_to_rgb(rgb_selected, depth_candidates, cfg)
    decision_mode = str(cfg.get("primary_pick_decision_mode", "depth_first")).strip().lower()

    decision_status = "no_pick"
    decision_reason = "rgb_and_depth_missing"
    decision_warning = None
    selected_candidate = None

    if decision_mode == "rgb_first_legacy":
        if rgb_selected is not None:
            if match_report["match_found"]:
                selected_candidate = dict(rgb_selected)
                selected_candidate["source"] = "rgb_seg"
                decision_status = "rgb_depth_agree_pass"
                decision_reason = "rgb_and_depth_matched_same_brick"
            elif rgb_quality["is_low_quality"] and depth_fallback["is_ready"]:
                selected_candidate = dict(depth_selected)
                selected_candidate["source"] = "depth_geom"
                decision_status = "depth_fallback_pass"
                decision_reason = "rgb_low_quality_depth_fallback"
            else:
                selected_candidate = dict(rgb_selected)
                selected_candidate["source"] = "rgb_seg"
                decision_status = "rgb_only_pass"
                if depth_selected is None:
                    decision_reason = "rgb_only_depth_missing"
                else:
                    decision_reason = "rgb_kept_without_depth_match"
                    decision_warning = "depth_mismatch"
        elif depth_fallback["is_ready"]:
            selected_candidate = dict(depth_selected)
            selected_candidate["source"] = "depth_geom"
            decision_status = "depth_fallback_pass"
            decision_reason = "rgb_missing_depth_fallback"
        elif depth_selected is not None:
            decision_reason = "rgb_missing_depth_not_ready"
            decision_warning = "depth_not_ready"
    else:
        if depth_fallback["is_ready"]:
            selected_candidate = dict(depth_selected)
            selected_candidate["source"] = "depth_geom"
            decision_status = "depth_primary_pass"
            if rgb_selected is None:
                decision_reason = "depth_primary_rgb_missing"
            elif match_report["match_found"]:
                decision_reason = "depth_primary_rgb_verified"
            elif rgb_quality["is_low_quality"]:
                decision_reason = "depth_primary_rgb_low_quality"
            else:
                decision_reason = "depth_primary_rgb_mismatch"
                decision_warning = "rgb_mismatch"
        elif rgb_selected is not None:
            selected_candidate = dict(rgb_selected)
            selected_candidate["source"] = "rgb_seg"
            decision_status = "rgb_fallback_pass"
            if depth_selected is None:
                decision_reason = "depth_missing_rgb_fallback"
            else:
                decision_reason = "depth_not_ready_rgb_fallback"
                decision_warning = "depth_not_ready"
        elif depth_selected is not None:
            decision_reason = "depth_not_ready_rgb_missing"
            decision_warning = "depth_not_ready"

    if selected_candidate is not None:
        selected_candidate["decision_status"] = decision_status
        selected_candidate["decision_reason"] = decision_reason
        selected_candidate["decision_warning"] = decision_warning
        selected_candidate["rgb_low_quality"] = bool(rgb_quality["is_low_quality"])
        selected_candidate["rgb_low_quality_reasons"] = list(rgb_quality["reasons"])
        selected_candidate["rgb_depth_match_found"] = bool(match_report["match_found"])
        selected_candidate["match_center_distance_px"] = None
        selected_candidate["match_angle_delta_deg"] = None
        selected_candidate["match_depth_delta_mm"] = None
        selected_candidate["match_bbox_iou"] = None
        selected_candidate["match_mask_iou"] = None
        if match_report["metrics"] is not None:
            selected_candidate["match_center_distance_px"] = float(match_report["metrics"]["center_distance_px"])
            selected_candidate["match_angle_delta_deg"] = float(match_report["metrics"]["angle_delta_deg"])
            selected_candidate["match_depth_delta_mm"] = float(match_report["metrics"]["depth_delta_mm"])
            selected_candidate["match_bbox_iou"] = float(match_report["metrics"]["bbox_iou"])
            selected_candidate["match_mask_iou"] = float(match_report["metrics"]["mask_iou"])
        if match_report["matched_candidate"] is not None:
            selected_candidate["matched_depth_geometry_score"] = _candidate_float(
                match_report["matched_candidate"],
                "geometry_score",
            )
        else:
            selected_candidate["matched_depth_geometry_score"] = None

    return {
        "decision_status": decision_status,
        "decision_reason": decision_reason,
        "decision_warning": decision_warning,
        "selected_candidate": selected_candidate,
        "rgb_selected_candidate": rgb_selected,
        "depth_selected_candidate": depth_selected,
        "matched_depth_candidate": match_report["matched_candidate"],
        "match_metrics": match_report["metrics"],
        "rgb_quality": rgb_quality,
        "depth_fallback": depth_fallback,
    }


def build_secondary_alignment_rgb_candidate(alignment_data: Optional[Sequence[float]]) -> Optional[Dict[str, object]]:
    if not isinstance(alignment_data, (list, tuple)) or len(alignment_data) < 2:
        return None
    try:
        pixel_x = float(alignment_data[0])
        pixel_y = float(alignment_data[1])
    except (TypeError, ValueError):
        return None
    return {
        "source": "rgb_legacy",
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "score": None,
        "geometry_score": None,
        "is_valid": True,
    }


def select_secondary_alignment_candidate(
    depth_report: Dict[str, object],
    rgb_alignment_data: Optional[Sequence[float]],
) -> Dict[str, object]:
    depth_candidate = depth_report.get("selected_candidate")
    rgb_candidate = build_secondary_alignment_rgb_candidate(rgb_alignment_data)

    selected_candidate = None
    decision_status = "secondary_no_pick"
    decision_reason = "secondary_depth_and_rgb_missing"
    rgb_fallback_used = False

    if isinstance(depth_candidate, dict) and bool(depth_candidate.get("is_valid")):
        selected_candidate = dict(depth_candidate)
        selected_candidate["decision_status"] = "secondary_depth_pass"
        selected_candidate["decision_reason"] = "secondary_depth_selected"
        selected_candidate["rgb_fallback_used"] = False
        decision_status = "secondary_depth_pass"
        decision_reason = "secondary_depth_selected"
    elif rgb_candidate is not None:
        selected_candidate = dict(rgb_candidate)
        selected_candidate["decision_status"] = "secondary_rgb_fallback_pass"
        selected_candidate["decision_reason"] = "secondary_depth_missing_rgb_fallback"
        selected_candidate["rgb_fallback_used"] = True
        decision_status = "secondary_rgb_fallback_pass"
        decision_reason = "secondary_depth_missing_rgb_fallback"
        rgb_fallback_used = True

    return {
        "decision_status": decision_status,
        "decision_reason": decision_reason,
        "selected_candidate": selected_candidate,
        "depth_candidate": depth_candidate,
        "rgb_alignment_result": rgb_candidate,
        "rgb_fallback_used": rgb_fallback_used,
    }


def _apply_mask_overlay(image: np.ndarray, mask: np.ndarray, color: tuple, alpha: float) -> np.ndarray:
    if not np.any(mask):
        return image
    overlay = image.copy()
    overlay[mask] = color
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)


def _draw_text_lines(image: np.ndarray, lines: List[str]) -> np.ndarray:
    for index, line in enumerate(lines):
        origin = (12, 24 + index * 22)
        cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def add_debug_banner(image: np.ndarray, lines: List[str]) -> np.ndarray:
    filtered_lines = [line for line in lines if line]
    if not filtered_lines:
        return np.ascontiguousarray(image)

    line_height = 24
    header_height = 18 + line_height * len(filtered_lines)
    panel = np.full((header_height + image.shape[0], image.shape[1], 3), 255, dtype=np.uint8)
    panel[header_height:, :, :] = image
    for index, line in enumerate(filtered_lines):
        origin = (14, 28 + index * line_height)
        cv2.putText(panel, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0, 0, 0), 2, cv2.LINE_AA)
    return np.ascontiguousarray(panel)


def build_primary_pick_decision_lines(fusion_report: Dict[str, object]) -> List[str]:
    lines: List[str] = []
    decision_status = str(fusion_report.get("decision_status"))
    selected_candidate = fusion_report.get("selected_candidate")
    source = "none"
    if isinstance(selected_candidate, dict):
        source = str(selected_candidate.get("source", "unknown"))
    lines.append(f"Decision {decision_status}  Source {source}")

    if isinstance(selected_candidate, dict):
        depth_text = "None"
        if selected_candidate.get("depth_mm") is not None:
            depth_text = f"{float(selected_candidate['depth_mm']):.1f}"
        angle_text = "None"
        if selected_candidate.get("angle_deg") is not None:
            angle_text = f"{float(selected_candidate['angle_deg']):.2f}"
        lines.append(
            f"Final ({selected_candidate.get('pixel_x')}, {selected_candidate.get('pixel_y')}) "
            f"A {angle_text}  Z {depth_text}"
        )

    match_metrics = fusion_report.get("match_metrics")
    if isinstance(match_metrics, dict):
        lines.append(
            f"Same-brick match dXY {float(match_metrics['center_distance_px']):.1f}  "
            f"dA {float(match_metrics['angle_delta_deg']):.2f}  "
            f"IoU {float(match_metrics['mask_iou']):.2f}  "
            f"dZ {float(match_metrics['depth_delta_mm']):.1f}"
        )
    elif fusion_report.get("rgb_selected_candidate") is not None and fusion_report.get("depth_selected_candidate") is not None:
        lines.append("Same-brick match: none")

    rgb_quality = fusion_report.get("rgb_quality")
    if isinstance(rgb_quality, dict) and bool(rgb_quality.get("is_low_quality")):
        lines.append(f"RGB low-quality: {','.join(rgb_quality.get('reasons') or [])}")

    depth_fallback = fusion_report.get("depth_fallback")
    if isinstance(depth_fallback, dict) and not bool(depth_fallback.get("is_ready")) and fusion_report.get("decision_status") == "no_pick":
        lines.append(f"Depth blocked: {','.join(depth_fallback.get('reasons') or [])}")

    warning = fusion_report.get("decision_warning")
    if warning:
        lines.append(f"Warning: {warning}")
    return lines


def build_secondary_alignment_decision_lines(decision_report: Dict[str, object]) -> List[str]:
    selected_candidate = decision_report.get("selected_candidate")
    source = "none"
    if isinstance(selected_candidate, dict):
        source = str(selected_candidate.get("source", "unknown"))

    lines = [
        f"Decision {decision_report.get('decision_status')}  Source {source}",
    ]
    if isinstance(selected_candidate, dict):
        angle_text = "None"
        if selected_candidate.get("angle_deg") is not None:
            angle_text = f"{float(selected_candidate['angle_deg']):.2f}"
        lines.append(
            f"Final ({int(round(float(selected_candidate['pixel_x'])))}, "
            f"{int(round(float(selected_candidate['pixel_y'])))})  A {angle_text}"
        )
        if selected_candidate.get("geometry_score") is not None:
            lines.append(f"Geom {float(selected_candidate['geometry_score']):.2f}")
    if bool(decision_report.get("rgb_fallback_used")):
        lines.append("RGB fallback used")
    elif decision_report.get("decision_status") == "secondary_depth_pass":
        lines.append("Depth candidate selected")
    elif decision_report.get("decision_status") == "secondary_no_pick":
        lines.append("No valid secondary alignment result")
    return lines


def render_debug_overlays(
    image: np.ndarray,
    depth_map: np.ndarray,
    candidate: Optional[Dict[str, object]],
    vision_debug: bool,
) -> tuple:
    rgb_debug = image.copy()

    if depth_map is None:
        depth_rgb = np.zeros_like(image)
    else:
        depth_view = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
        depth_8bit = cv2.normalize(depth_view, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        depth_bgr = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
        depth_rgb = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)

    if candidate is None:
        return rgb_debug, depth_rgb

    inlier_mask = candidate["inlier_mask"]
    rejected_mask = candidate["rejected_mask"]
    if vision_debug:
        rgb_debug = _apply_mask_overlay(rgb_debug, rejected_mask, (0, 0, 255), 0.30)
        rgb_debug = _apply_mask_overlay(rgb_debug, inlier_mask, (0, 255, 0), 0.30)
        depth_rgb = _apply_mask_overlay(depth_rgb, rejected_mask, (255, 0, 0), 0.30)
        depth_rgb = _apply_mask_overlay(depth_rgb, inlier_mask, (0, 255, 0), 0.30)

    if candidate["pixel_x"] is not None and candidate["pixel_y"] is not None:
        center = (int(candidate["pixel_x"]), int(candidate["pixel_y"]))
        cv2.circle(rgb_debug, center, 5, (0, 255, 255), -1)
        cv2.circle(depth_rgb, center, 5, (255, 255, 255), -1)

        line_length = max(40, int(math.sqrt(max(candidate["mask_foreground_count"], 1))))
        angle_rad = math.radians(float(candidate["angle_deg"]))
        end_point = (
            int(round(center[0] + math.cos(angle_rad) * line_length)),
            int(round(center[1] + math.sin(angle_rad) * line_length)),
        )
        cv2.line(rgb_debug, center, end_point, (0, 0, 255), 2)
        cv2.line(depth_rgb, center, end_point, (255, 255, 255), 2)

    text_lines = [
        (
            f"Xpx {candidate['pixel_x']} Ypx {candidate['pixel_y']} "
            f"Zmm {candidate['depth_mm']:.1f}"
            if candidate["depth_mm"] is not None
            else f"Xpx {candidate['pixel_x']} Ypx {candidate['pixel_y']} Zmm None"
        ),
        (
            f"Angle {candidate['angle_deg']:.2f} "
            f"RawPCA {candidate['raw_pca_angle_deg']:.2f} "
            f"Score {candidate['score']:.3f}"
        ),
        (
            f"ValidN {candidate['valid_depth_count']} "
            f"ValidR {candidate['valid_depth_ratio']:.3f}"
        ),
    ]
    if candidate["angle_fallback"]:
        text_lines.append("ANGLE_FALLBACK")

    rgb_debug = _draw_text_lines(rgb_debug, text_lines)
    depth_rgb = _draw_text_lines(depth_rgb, text_lines)
    return rgb_debug, depth_rgb


class Blinx_image_rec:
    def __init__(self, public_class):
        self.public_class = public_class
        self.image_camera = Blinx_Image_Camera()

        self.model_path = "ImageProcessing/gangjin2-m.onnx"
        self.detection = YOLOv8(self.model_path)
        self.session, self.model_inputs = self.detection.init_detect_model()

        self.model_path1 = "ImageProcessing/hongzhuan-detect.onnx"
        self.detection1 = YOLOv82(self.model_path1)
        self.session1, self.model_inputs1 = self.detection1.init_detect_model()

        self.model_path2 = "ImageProcessing/hongzhuan-seg2.onnx"
        self.model2 = YOLOv8Seg(self.model_path2)
        self.conf = 0.65
        self.iou = 0.45
        self.rgbd_cfg = {
            "depth_min_valid_mm": float(getattr(self.public_class, "depth_min_valid_mm", 10.0)),
            "depth_trim_percent": float(getattr(self.public_class, "depth_trim_percent", 5.0)),
            "depth_min_valid_pixels": int(getattr(self.public_class, "depth_min_valid_pixels", 200)),
            "depth_min_valid_ratio": float(getattr(self.public_class, "depth_min_valid_ratio", 0.2)),
            "depth_erode_kernel": int(getattr(self.public_class, "depth_erode_kernel", 3)),
            "depth_erode_iterations": int(getattr(self.public_class, "depth_erode_iterations", 1)),
            "pca_min_axis_ratio": float(getattr(self.public_class, "pca_min_axis_ratio", 1.10)),
            "vision_debug": bool(int(getattr(self.public_class, "vision_debug", 1))),
        }
        self.depth_geom_cfg = {
            "primary_pick_depth_roi": getattr(self.public_class, "primary_pick_depth_roi", [0, 0, 0, 0]),
            "depth_min_valid_mm": float(getattr(self.public_class, "depth_min_valid_mm", 10.0)),
            "depth_geom_board_estimation_mode": getattr(
                self.public_class,
                "depth_geom_board_estimation_mode",
                "global_hist",
            ),
            "depth_geom_hist_bin_mm": float(getattr(self.public_class, "depth_geom_hist_bin_mm", 2.0)),
            "depth_geom_hist_peak_refine_window_mm": float(
                getattr(self.public_class, "depth_geom_hist_peak_refine_window_mm", 3.0)
            ),
            "depth_geom_expected_long_mm": float(getattr(self.public_class, "depth_geom_expected_long_mm", 200.0)),
            "depth_geom_expected_short_mm": float(getattr(self.public_class, "depth_geom_expected_short_mm", 100.0)),
            "depth_geom_expected_height_mm": float(
                getattr(self.public_class, "depth_geom_expected_height_mm", 70.0)
            ),
            "depth_geom_long_tol_mm": float(getattr(self.public_class, "depth_geom_long_tol_mm", 40.0)),
            "depth_geom_short_tol_mm": float(getattr(self.public_class, "depth_geom_short_tol_mm", 30.0)),
            "depth_geom_height_tol_mm": float(getattr(self.public_class, "depth_geom_height_tol_mm", 20.0)),
            "depth_geom_min_brick_height_mm": float(
                getattr(self.public_class, "depth_geom_min_brick_height_mm", 25.0)
            ),
            "depth_geom_max_brick_height_mm": float(
                getattr(self.public_class, "depth_geom_max_brick_height_mm", 130.0)
            ),
            "depth_geom_min_region_area_px": int(
                getattr(self.public_class, "depth_geom_min_region_area_px", 2000)
            ),
            "depth_geom_max_region_area_px": int(
                getattr(self.public_class, "depth_geom_max_region_area_px", 400000)
            ),
            "depth_geom_median_kernel": int(getattr(self.public_class, "depth_geom_median_kernel", 3)),
            "depth_geom_open_kernel": int(getattr(self.public_class, "depth_geom_open_kernel", 3)),
            "depth_geom_close_kernel": int(getattr(self.public_class, "depth_geom_close_kernel", 5)),
            "depth_geom_border_margin_px": int(getattr(self.public_class, "depth_geom_border_margin_px", 12)),
            "depth_geom_planarity_max_std_mm": float(
                getattr(self.public_class, "depth_geom_planarity_max_std_mm", 3.0)
            ),
            "depth_geom_min_rectangularity": float(
                getattr(self.public_class, "depth_geom_min_rectangularity", 0.65)
            ),
            "depth_geom_min_completeness": float(
                getattr(self.public_class, "depth_geom_min_completeness", 0.60)
            ),
        }
        self.secondary_depth_geom_cfg = dict(self.depth_geom_cfg)
        self.secondary_depth_geom_cfg.update(
            {
                "primary_pick_depth_roi": getattr(
                    self.public_class,
                    "secondary_alignment_depth_roi",
                    self.depth_geom_cfg["primary_pick_depth_roi"],
                ),
                "depth_geom_min_region_area_px": int(
                    getattr(
                        self.public_class,
                        "secondary_depth_geom_min_region_area_px",
                        self.depth_geom_cfg["depth_geom_min_region_area_px"],
                    )
                ),
                "depth_geom_max_region_area_px": int(
                    getattr(
                        self.public_class,
                        "secondary_depth_geom_max_region_area_px",
                        self.depth_geom_cfg["depth_geom_max_region_area_px"],
                    )
                ),
            }
        )
        self.primary_pick_fusion_cfg = {
            "primary_pick_rgb_depth_center_thresh_px": float(
                getattr(self.public_class, "primary_pick_rgb_depth_center_thresh_px", 80.0)
            ),
            "primary_pick_rgb_depth_angle_thresh_deg": float(
                getattr(self.public_class, "primary_pick_rgb_depth_angle_thresh_deg", 20.0)
            ),
            "primary_pick_rgb_depth_iou_thresh": float(
                getattr(self.public_class, "primary_pick_rgb_depth_iou_thresh", 0.20)
            ),
            "primary_pick_rgb_depth_mm_thresh": float(
                getattr(self.public_class, "primary_pick_rgb_depth_mm_thresh", 10.0)
            ),
            "primary_pick_rgb_low_score_thresh": float(
                getattr(self.public_class, "primary_pick_rgb_low_score_thresh", 0.90)
            ),
            "primary_pick_rgb_low_valid_ratio_thresh": float(
                getattr(self.public_class, "primary_pick_rgb_low_valid_ratio_thresh", 0.70)
            ),
            "primary_pick_depth_fallback_geom_thresh": float(
                getattr(self.public_class, "primary_pick_depth_fallback_geom_thresh", 0.88)
            ),
            "primary_pick_decision_mode": getattr(
                self.public_class,
                "primary_pick_decision_mode",
                "depth_first",
            ),
        }

    def blinx_rebar_image_rec(self, image):
        output_image, result_list = self.detection.detect(self.session, self.model_inputs, image)
        print(result_list)
        data = []
        for result in result_list:
            data.append([result[2], result[3]])
        return output_image, data

    def blinx_brickandporcelain_image_rec(self, image):
        output_image, result_list = self.detection1.detect(self.session1, self.model_inputs1, image)
        print(result_list)
        if not result_list:
            return output_image, None
        data = [result_list[0][2], result_list[0][3]]
        return output_image, data

    def _build_rgbd_segmentation_report(self, image, mech_depth_map):
        boxes, segments, masks = self.model2(image, conf_threshold=self.conf, iou_threshold=self.iou)
        if len(boxes) == 0:
            rgb_debug, depth_debug = render_debug_overlays(
                image,
                mech_depth_map,
                None,
                self.rgbd_cfg["vision_debug"],
            )
            return {
                "rgb_debug": rgb_debug,
                "depth_debug": depth_debug,
                "candidates": [],
                "ranked_candidates": [],
                "selected_candidate": None,
                "selected_rank_index": None,
            }

        candidates = []
        instances = self.model2.build_instance_records(boxes, segments, masks)
        for instance in instances:
            mask = instance["mask"].astype(bool)
            depth_stats = extract_depth_stats(mask, mech_depth_map, self.rgbd_cfg)
            angle_stats = compute_pca_angle(mask, self.rgbd_cfg)
            candidate = {
                "source": "rgb_seg",
                "class_id": instance["class_id"],
                "score": instance["score"],
                "bbox": instance["bbox"],
                "segment": instance["segment"],
                "mask": mask,
                "mask_foreground_count": int(mask.sum()),
                "pixel_x": depth_stats["pixel_x"],
                "pixel_y": depth_stats["pixel_y"],
                "depth_mm": depth_stats["depth_mm"],
                "angle_deg": angle_stats["angle_deg"],
                "raw_pca_angle_deg": angle_stats["raw_pca_angle_deg"],
                "valid_depth_count": depth_stats["valid_depth_count"],
                "valid_depth_ratio": depth_stats["valid_depth_ratio"],
                "inlier_mask": depth_stats["inlier_mask"],
                "rejected_mask": depth_stats["rejected_mask"],
                "angle_fallback": angle_stats["angle_fallback"],
                "axis_ratio": angle_stats["axis_ratio"],
                "is_valid": depth_stats["is_valid"],
            }
            candidates.append(candidate)

        ranked_candidates = rank_candidates(candidates)
        selected_candidate = next((item for item in ranked_candidates if item["is_valid"]), None)
        selected_rank_index = None
        if selected_candidate is not None:
            for index, item in enumerate(ranked_candidates):
                if item is selected_candidate:
                    selected_rank_index = index
                    break
        debug_candidate = selected_candidate or (ranked_candidates[0] if ranked_candidates else None)
        rgb_debug, depth_debug = render_debug_overlays(
            image,
            mech_depth_map,
            debug_candidate,
            self.rgbd_cfg["vision_debug"],
        )
        return {
            "rgb_debug": rgb_debug,
            "depth_debug": depth_debug,
            "candidates": candidates,
            "ranked_candidates": ranked_candidates,
            "selected_candidate": selected_candidate,
            "selected_rank_index": selected_rank_index,
        }

    def _run_rgbd_segmentation(self, image, mech_depth_map):
        report = self._build_rgbd_segmentation_report(image, mech_depth_map)
        return report["rgb_debug"], report["depth_debug"], report["selected_candidate"]

    def _build_depth_geometry_report(self, image, mech_depth_map, depth_geom_cfg=None):
        depth_geom_cfg = self.depth_geom_cfg if depth_geom_cfg is None else depth_geom_cfg
        preprocess_result = preprocess_depth_roi(mech_depth_map, depth_geom_cfg)
        height_map = build_height_map(preprocess_result["board_depth_mm"], preprocess_result["depth_filtered"])
        foreground_mask = extract_depth_foreground(height_map, preprocess_result["valid_mask"], depth_geom_cfg)
        regions = split_connected_regions(foreground_mask, height_map, preprocess_result["valid_mask"], depth_geom_cfg)

        candidates = []
        for region in regions:
            rectangle_fit = fit_region_rectangle(region["mask"], depth_geom_cfg)
            if rectangle_fit is None:
                continue
            candidate = build_depth_candidate(
                region,
                rectangle_fit,
                preprocess_result,
                depth_geom_cfg,
                self.image_camera,
                mech_depth_map.shape,
            )
            candidates.append(candidate)

        ranked_candidates = rank_depth_candidates(candidates)
        selected_candidate = next((item for item in ranked_candidates if item["is_valid"]), None)
        selected_rank_index = None
        if selected_candidate is not None:
            for index, item in enumerate(ranked_candidates):
                if item is selected_candidate:
                    selected_rank_index = index
                    break

        debug_images = render_depth_debug_panel(
            image,
            mech_depth_map,
            preprocess_result,
            height_map,
            foreground_mask,
            regions,
            ranked_candidates,
            selected_candidate,
        )
        return {
            "rgb_debug": debug_images["selected_overlay"],
            "depth_debug": debug_images["analysis_panel"],
            "debug_images": debug_images,
            "candidates": candidates,
            "ranked_candidates": ranked_candidates,
            "selected_candidate": selected_candidate,
            "selected_rank_index": selected_rank_index,
            "board_depth_mm": preprocess_result["board_depth_mm"],
            "workspace_roi": list(preprocess_result["workspace_roi"]),
            "render_state": {
                "preprocess_result": preprocess_result,
                "height_map": height_map,
                "foreground_mask": foreground_mask,
                "regions": regions,
            },
        }

    def _build_primary_pick_fusion_report(self, image, mech_depth_map):
        rgb_report = self._build_rgbd_segmentation_report(image, mech_depth_map)
        depth_report = self._build_depth_geometry_report(image, mech_depth_map)
        fusion_report = select_primary_pick_candidate(
            rgb_report,
            depth_report,
            self.primary_pick_fusion_cfg,
        )

        rgb_debug = np.ascontiguousarray(rgb_report["rgb_debug"])
        depth_debug = np.ascontiguousarray(depth_report["depth_debug"])
        matched_depth_candidate = fusion_report.get("matched_depth_candidate")
        if (
            fusion_report["decision_status"] == "rgb_depth_agree_pass"
            and matched_depth_candidate is not None
            and matched_depth_candidate is not depth_report.get("selected_candidate")
        ):
            render_state = depth_report.get("render_state") or {}
            debug_images = render_depth_debug_panel(
                image,
                mech_depth_map,
                render_state["preprocess_result"],
                render_state["height_map"],
                render_state["foreground_mask"],
                render_state["regions"],
                depth_report["ranked_candidates"],
                matched_depth_candidate,
            )
            depth_report["debug_images"] = debug_images
            depth_report["depth_debug"] = debug_images["analysis_panel"]
            depth_report["rgb_debug"] = debug_images["selected_overlay"]
            depth_debug = np.ascontiguousarray(depth_report["depth_debug"])

        decision_lines = build_primary_pick_decision_lines(fusion_report)
        rgb_debug = add_debug_banner(rgb_debug, decision_lines)
        depth_debug = add_debug_banner(depth_debug, decision_lines)
        return {
            "rgb_debug": rgb_debug,
            "depth_debug": depth_debug,
            "selected_candidate": fusion_report["selected_candidate"],
            "fusion_report": fusion_report,
            "rgb_report": rgb_report,
            "depth_report": depth_report,
        }

    def _build_secondary_alignment_depth_first_report(self, image, mech_depth_map):
        rgb_debug, rgb_alignment_data = self.blinx_brickandporcelain_image_rec(image)
        depth_report = self._build_depth_geometry_report(image, mech_depth_map, self.secondary_depth_geom_cfg)
        decision_report = select_secondary_alignment_candidate(depth_report, rgb_alignment_data)

        selected_candidate = decision_report.get("selected_candidate")
        if decision_report["decision_status"] == "secondary_depth_pass":
            rgb_display = np.ascontiguousarray(depth_report["debug_images"].get("rectangles", image.copy()))
            if isinstance(selected_candidate, dict):
                center = (
                    int(round(float(selected_candidate["pixel_x"]))),
                    int(round(float(selected_candidate["pixel_y"]))),
                )
                cv2.circle(rgb_display, center, 6, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.putText(
                    rgb_display,
                    f"Depth XY ({center[0]}, {center[1]})",
                    (center[0] + 12, max(center[1] - 14, 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
        else:
            rgb_display = np.ascontiguousarray(rgb_debug)

        depth_debug = np.ascontiguousarray(depth_report["depth_debug"])
        decision_lines = build_secondary_alignment_decision_lines(decision_report)
        rgb_display = add_debug_banner(rgb_display, decision_lines)
        depth_debug = add_debug_banner(depth_debug, decision_lines)
        return {
            "rgb_debug": rgb_display,
            "depth_debug": depth_debug,
            "selected_candidate": selected_candidate,
            "decision_report": decision_report,
            "depth_report": depth_report,
            "rgb_alignment_data": rgb_alignment_data,
        }

    def blinx_brick_image_rec_debug(self, image, mech_depth_map):
        return self._build_rgbd_segmentation_report(image, mech_depth_map)

    def blinx_brick_image_rec(self, image, mech_depth_map):
        return self._run_rgbd_segmentation(image, mech_depth_map)

    def blinx_brick_image_rec2(self, image, mech_depth_map):
        return self._run_rgbd_segmentation(image, mech_depth_map)

    def blinx_brick_depth_candidates(self, image, mech_depth_map):
        return self._build_depth_geometry_report(image, mech_depth_map)

    def blinx_brick_secondary_depth_candidates(self, image, mech_depth_map):
        return self._build_depth_geometry_report(image, mech_depth_map, self.secondary_depth_geom_cfg)

    def blinx_brick_primary_pick_fusion(self, image, mech_depth_map):
        report = self._build_primary_pick_fusion_report(image, mech_depth_map)
        return report["rgb_debug"], report["depth_debug"], report["selected_candidate"], report

    def blinx_brick_secondary_alignment_depth_first(self, image, mech_depth_map):
        report = self._build_secondary_alignment_depth_first_report(image, mech_depth_map)
        return report["rgb_debug"], report["depth_debug"], report["selected_candidate"], report

    def blinx_image_gain(self, img):
        try:
            orig_img = img.copy()
            scale_percent = 25
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rectangles_info = []
            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) != 4:
                    continue

                area = cv2.contourArea(approx)
                if area < 100 or area > (width * height * 0.5):
                    continue
                if not cv2.isContourConvex(approx):
                    continue

                rect = cv2.minAreaRect(approx)
                (cx, cy), (box_w, box_h), angle = rect
                aspect_ratio = max(box_w, box_h) / (min(box_w, box_h) + 1e-5)
                if not (0.7 < aspect_ratio < 1.4 and area > 1000):
                    continue

                scale_factor = 100 / scale_percent
                cx_orig = int(cx * scale_factor)
                cy_orig = int(cy * scale_factor)
                box = cv2.boxPoints(((cx_orig, cy_orig), (box_w * scale_factor, box_h * scale_factor), angle))
                box = np.intp(box)
                rectangles_info.append(
                    {
                        "center": (cx_orig, cy_orig),
                        "angle": angle,
                        "contour": box,
                        "size": (int(box_w * scale_factor), int(box_h * scale_factor)),
                    }
                )

            for info in rectangles_info:
                cv2.drawContours(orig_img, [info["contour"]], 0, (0, 255, 0), 2)
                cv2.circle(orig_img, info["center"], 5, (0, 0, 255), -1)
                angle_rad = math.radians(info["angle"])
                end_x = int(info["center"][0] + 50 * math.cos(angle_rad))
                end_y = int(info["center"][1] + 50 * math.sin(angle_rad))
                cv2.line(orig_img, info["center"], (end_x, end_y), (255, 0, 0), 2)
                cv2.putText(
                    orig_img,
                    f"Angle: {info['angle']:.1f}",
                    (info["center"][0] + 10, info["center"][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )

            if not rectangles_info:
                return orig_img, None, None, None
            return (
                orig_img,
                rectangles_info[0]["center"][0],
                rectangles_info[0]["center"][1],
                rectangles_info[0]["angle"],
            )
        except Exception as exc:
            print(exc)
            return img, None, None, None

    def blinx_image_gain2(self, image):
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        roi = (212, 60, 939, 484)
        x, y, width, height = roi
        roi_image = image[y : y + height, x : x + width]

        roi_image = cv2.GaussianBlur(roi_image, (5, 5), 0)
        roi_image = cv2.medianBlur(roi_image, 5)
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100000:
                continue

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            box[:, 0] += x
            box[:, 1] += y

            center = (int(rect[0][0] + x), int(rect[0][1] + y))
            angle = rect[2]
            cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
            cv2.circle(image, center, 5, (0, 255, 0), -1)
            cv2.putText(
                image,
                f"Angle: {angle:.1f} deg",
                (center[0], center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
            return image, center[0], center[1], angle

        return image, None, None, None
