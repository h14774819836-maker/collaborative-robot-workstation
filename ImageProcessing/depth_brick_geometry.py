from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def normalize_angle_90(angle_deg: float) -> float:
    return ((float(angle_deg) + 90.0) % 180.0) - 90.0


def calculate_rectangle_angle(corners: np.ndarray) -> float:
    corners = np.asarray(corners, dtype=np.float32)
    if corners.shape[0] != 4:
        raise ValueError("four corner points are required")

    edges: list[tuple[float, tuple[np.ndarray, np.ndarray]]] = []
    for index in range(4):
        p1 = corners[index]
        p2 = corners[(index + 1) % 4]
        edge_length = float(np.linalg.norm(p1 - p2))
        edges.append((edge_length, (p1, p2)))

    edges.sort(key=lambda item: item[0], reverse=True)
    p1, p2 = edges[0][1]
    vector = p2 - p1
    angle_rad = np.arctan2(vector[1], vector[0])
    return normalize_angle_90(np.degrees(angle_rad))


def _ensure_odd_kernel(value: Any, *, minimum: int = 1) -> int:
    kernel = max(int(value), int(minimum))
    if kernel % 2 == 0:
        kernel += 1
    return kernel


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_roi(roi: Any, image_shape: Sequence[int]) -> Tuple[int, int, int, int]:
    height, width = int(image_shape[0]), int(image_shape[1])
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        return 0, 0, width, height

    try:
        x1, y1, x2, y2 = [int(round(float(value))) for value in roi]
    except (TypeError, ValueError):
        return 0, 0, width, height

    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0, 0, width, height
    return x1, y1, x2, y2


def _depth_to_preview(depth_map: np.ndarray) -> np.ndarray:
    view = np.nan_to_num(np.asarray(depth_map, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    depth_8bit = cv2.normalize(view, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    depth_bgr = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
    return cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)


def _gray_to_rgb(gray_image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.asarray(gray_image, dtype=np.uint8), cv2.COLOR_GRAY2RGB)


def _pad_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.shape[0] == target_height:
        return image
    pad_bottom = max(target_height - image.shape[0], 0)
    return cv2.copyMakeBorder(
        image,
        0,
        pad_bottom,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def _build_panel_cell(title: str, image: np.ndarray) -> np.ndarray:
    header_height = 48
    cell = np.full((header_height + image.shape[0], image.shape[1], 3), 255, dtype=np.uint8)
    cell[header_height:, :, :] = image
    cv2.putText(cell, title, (14, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(cell, title, (14, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 2, cv2.LINE_AA)
    return cell


def _draw_multiline_label(
    image: np.ndarray,
    anchor: tuple[int, int],
    lines: Sequence[str],
    *,
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    font_scale: float = 0.72,
    thickness: int = 2,
) -> None:
    lines = [line for line in lines if line]
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    padding_x = 10
    padding_y = 10
    line_gap = 10
    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    max_width = max(size[0] for size in text_sizes)
    line_height = max(size[1] for size in text_sizes)
    block_height = padding_y * 2 + len(lines) * line_height + max(0, len(lines) - 1) * line_gap
    block_width = padding_x * 2 + max_width

    x = int(np.clip(anchor[0], 8, max(image.shape[1] - block_width - 8, 8)))
    y = int(np.clip(anchor[1], 8, max(image.shape[0] - block_height - 8, 8)))

    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + block_width, y + block_height), bg_color, -1)
    image[:] = cv2.addWeighted(overlay, 0.56, image, 0.44, 0.0)
    cv2.rectangle(image, (x, y), (x + block_width, y + block_height), (255, 255, 255), 2)

    baseline_y = y + padding_y + line_height
    for index, line in enumerate(lines):
        origin = (x + padding_x, baseline_y + index * (line_height + line_gap))
        cv2.putText(image, line, origin, font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(image, line, origin, font, font_scale, text_color, thickness, cv2.LINE_AA)


def _draw_angle_indicator(
    image: np.ndarray,
    center: tuple[int, int],
    angle_deg: float,
    *,
    color: tuple[int, int, int] = (255, 255, 255),
    length: int = 90,
) -> None:
    angle_rad = np.deg2rad(float(angle_deg))
    dx = int(round(np.cos(angle_rad) * length))
    dy = int(round(np.sin(angle_rad) * length))
    end_point = (center[0] + dx, center[1] + dy)
    cv2.arrowedLine(image, center, end_point, (0, 0, 0), 6, cv2.LINE_AA, tipLength=0.16)
    cv2.arrowedLine(image, center, end_point, color, 3, cv2.LINE_AA, tipLength=0.16)


def _apply_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    *,
    alpha: float,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return image.copy()
    overlay = image.copy()
    overlay[mask] = color
    return cv2.addWeighted(overlay, float(alpha), image, 1.0 - float(alpha), 0.0)


def _draw_mask_outline(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
) -> None:
    mask_u8 = np.asarray(mask, dtype=np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image, contours, -1, color, thickness, cv2.LINE_AA)


def _stack_panel_rows(rows: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
    rendered_rows: list[np.ndarray] = []
    for row in rows:
        if not row:
            continue
        target_height = max(cell.shape[0] for cell in row)
        padded = [_pad_to_height(cell, target_height) for cell in row]
        separator = np.full((target_height, 16, 3), 255, dtype=np.uint8)
        content = padded[0]
        for cell in padded[1:]:
            content = np.hstack((content, separator, cell))
        rendered_rows.append(content)

    if not rendered_rows:
        return np.full((1, 1, 3), 255, dtype=np.uint8)

    target_width = max(row.shape[1] for row in rendered_rows)
    padded_rows = []
    for row in rendered_rows:
        if row.shape[1] < target_width:
            pad_right = target_width - row.shape[1]
            row = cv2.copyMakeBorder(
                row,
                0,
                0,
                0,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )
        padded_rows.append(row)
    separator = np.full((18, target_width, 3), 255, dtype=np.uint8)
    panel = padded_rows[0]
    for row in padded_rows[1:]:
        panel = np.vstack((panel, separator, row))
    return panel


def estimate_board_depth_global_hist(valid_depth_values: np.ndarray, cfg: Dict[str, Any]) -> Optional[float]:
    values = np.asarray(valid_depth_values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None

    bin_mm = max(_coerce_float(cfg.get("depth_geom_hist_bin_mm"), 2.0), 0.1)
    refine_window_mm = max(_coerce_float(cfg.get("depth_geom_hist_peak_refine_window_mm"), 3.0), 0.1)

    min_depth = float(np.min(values))
    max_depth = float(np.max(values))
    if max_depth - min_depth <= bin_mm:
        return float(np.median(values))

    bins = np.arange(min_depth, max_depth + bin_mm, bin_mm, dtype=np.float32)
    if bins.size < 2:
        return float(np.median(values))

    hist, edges = np.histogram(values, bins=bins)
    peak_index = int(np.argmax(hist))
    peak_center = float((edges[peak_index] + edges[peak_index + 1]) * 0.5)
    refine_mask = (values >= peak_center - refine_window_mm) & (values <= peak_center + refine_window_mm)
    refined_values = values[refine_mask]
    if refined_values.size == 0:
        lower = float(edges[peak_index])
        upper = float(edges[peak_index + 1])
        refined_values = values[(values >= lower) & (values <= upper)]
    if refined_values.size == 0:
        refined_values = values
    return float(np.median(refined_values))


def preprocess_depth_roi(depth_map: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    depth_map = np.asarray(depth_map, dtype=np.float32)
    if depth_map.ndim != 2:
        raise ValueError("Depth map must have shape (height, width)")

    x1, y1, x2, y2 = _coerce_roi(cfg.get("primary_pick_depth_roi"), depth_map.shape)
    depth_roi = depth_map[y1:y2, x1:x2].copy()
    valid_mask = np.isfinite(depth_roi) & (depth_roi > _coerce_float(cfg.get("depth_min_valid_mm"), 10.0))
    valid_values = depth_roi[valid_mask]
    board_depth_mm = estimate_board_depth_global_hist(valid_values, cfg)

    filled_roi = depth_roi.copy()
    if board_depth_mm is not None:
        filled_roi[~valid_mask] = float(board_depth_mm)
    else:
        filled_roi[~valid_mask] = 0.0

    median_kernel = _ensure_odd_kernel(cfg.get("depth_geom_median_kernel", 3))
    if median_kernel > 1:
        filtered_roi = cv2.medianBlur(filled_roi.astype(np.float32), median_kernel)
    else:
        filtered_roi = filled_roi.astype(np.float32)
    filtered_roi = filtered_roi.astype(np.float32)
    filtered_roi[~valid_mask] = np.nan

    return {
        "workspace_roi": (x1, y1, x2, y2),
        "depth_roi": depth_roi,
        "valid_mask": valid_mask,
        "valid_depth_count": int(np.count_nonzero(valid_mask)),
        "board_depth_mm": board_depth_mm,
        "depth_filtered": filtered_roi,
    }


def build_height_map(board_depth_mm: Optional[float], depth_filtered: np.ndarray) -> np.ndarray:
    height_map = np.full_like(np.asarray(depth_filtered, dtype=np.float32), np.nan, dtype=np.float32)
    if board_depth_mm is None:
        return height_map
    valid_mask = np.isfinite(depth_filtered)
    height_map[valid_mask] = float(board_depth_mm) - depth_filtered[valid_mask]
    return height_map


def extract_depth_foreground(height_map: np.ndarray, valid_mask: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    min_height = _coerce_float(cfg.get("depth_geom_min_brick_height_mm"), 20.0)
    max_height = _coerce_float(cfg.get("depth_geom_max_brick_height_mm"), 200.0)
    foreground = (
        np.asarray(valid_mask, dtype=bool)
        & np.isfinite(height_map)
        & (height_map >= min_height)
        & (height_map <= max_height)
    )

    open_kernel = _ensure_odd_kernel(cfg.get("depth_geom_open_kernel", 3))
    close_kernel = _ensure_odd_kernel(cfg.get("depth_geom_close_kernel", 5))
    foreground_u8 = foreground.astype(np.uint8)
    if open_kernel > 1:
        open_struct = np.ones((open_kernel, open_kernel), dtype=np.uint8)
        foreground_u8 = cv2.morphologyEx(foreground_u8, cv2.MORPH_OPEN, open_struct)
    if close_kernel > 1:
        close_struct = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        foreground_u8 = cv2.morphologyEx(foreground_u8, cv2.MORPH_CLOSE, close_struct)
    return foreground_u8.astype(bool)


def split_connected_regions(foreground_mask: np.ndarray, height_map: np.ndarray, valid_mask: np.ndarray, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    foreground_u8 = np.asarray(foreground_mask, dtype=np.uint8)
    if np.count_nonzero(foreground_u8) == 0:
        return []

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground_u8, connectivity=8)
    min_area = int(max(_coerce_float(cfg.get("depth_geom_min_region_area_px"), 100.0), 1.0))
    max_area = int(max(_coerce_float(cfg.get("depth_geom_max_region_area_px"), 1e9), float(min_area)))
    border_margin = int(max(_coerce_float(cfg.get("depth_geom_border_margin_px"), 12.0), 0.0))
    max_std = _coerce_float(cfg.get("depth_geom_planarity_max_std_mm"), 3.0)
    roi_height, roi_width = foreground_u8.shape

    regions: list[dict[str, Any]] = []
    for label_index in range(1, num_labels):
        area_px = int(stats[label_index, cv2.CC_STAT_AREA])
        if area_px < min_area or area_px > max_area:
            continue

        region_mask = labels == label_index
        region_heights = height_map[region_mask & np.isfinite(height_map)]
        if region_heights.size == 0:
            continue

        height_std_mm = float(np.std(region_heights))
        if height_std_mm > max_std:
            continue

        x = int(stats[label_index, cv2.CC_STAT_LEFT])
        y = int(stats[label_index, cv2.CC_STAT_TOP])
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        touch_border = (
            x <= border_margin
            or y <= border_margin
            or (x + width) >= (roi_width - border_margin)
            or (y + height) >= (roi_height - border_margin)
        )
        valid_count = int(np.count_nonzero(region_mask & valid_mask))
        regions.append(
            {
                "label": label_index,
                "mask": region_mask,
                "area_px": area_px,
                "bbox": [x, y, x + width, y + height],
                "touch_border": touch_border,
                "centroid_px": (float(centroids[label_index][0]), float(centroids[label_index][1])),
                "mean_height_mm": float(np.mean(region_heights)),
                "median_height_mm": float(np.median(region_heights)),
                "height_std_mm": height_std_mm,
                "valid_depth_count": valid_count,
                "valid_depth_ratio": float(valid_count / max(area_px, 1)),
            }
        )
    return regions


def fit_region_rectangle(region_mask: np.ndarray, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    region_mask = np.asarray(region_mask, dtype=bool)
    area_px = int(np.count_nonzero(region_mask))
    if area_px < 4:
        return None

    erode_kernel = np.ones((3, 3), dtype=np.uint8)
    eroded_mask = cv2.erode(region_mask.astype(np.uint8), erode_kernel, iterations=1).astype(bool)
    if np.count_nonzero(eroded_mask) < int(area_px * 0.6):
        fit_mask = region_mask
    else:
        fit_mask = eroded_mask

    ys, xs = np.where(fit_mask)
    if xs.size < 2:
        return None

    points = np.column_stack((xs, ys)).astype(np.float32)
    rect = cv2.minAreaRect(points)
    box_points = cv2.boxPoints(rect).astype(np.float32)
    width_px, height_px = rect[1]
    long_side_px = float(max(width_px, height_px))
    short_side_px = float(min(width_px, height_px))
    rect_area_px = float(max(width_px * height_px, 1.0))

    region_points = np.column_stack((np.where(region_mask)[1], np.where(region_mask)[0])).astype(np.float32)
    hull = cv2.convexHull(region_points)
    hull_area_px = float(max(cv2.contourArea(hull), 1.0))
    region_area_px = float(max(area_px, 1.0))
    rectangularity_score = float(np.clip(hull_area_px / rect_area_px, 0.0, 1.0))
    completeness_score = float(np.clip(region_area_px / hull_area_px, 0.0, 1.0))

    return {
        "fit_mask": fit_mask,
        "center_x": float(rect[0][0]),
        "center_y": float(rect[0][1]),
        "width_px": float(width_px),
        "height_px": float(height_px),
        "long_side_px": long_side_px,
        "short_side_px": short_side_px,
        "angle_deg": calculate_rectangle_angle(box_points),
        "box_points": box_points,
        "rect_area_px": rect_area_px,
        "rectangularity_score": rectangularity_score,
        "completeness_score": completeness_score,
    }


def measure_rectangle_mm(box_points: np.ndarray, depth_mm: float, image_camera: Any) -> Dict[str, Any]:
    box_points = np.asarray(box_points, dtype=np.float32)
    if box_points.shape != (4, 2):
        raise ValueError("box_points must have shape (4, 2)")

    z_values = np.full(4, float(depth_mm), dtype=np.float32)
    xs = box_points[:, 0]
    ys = box_points[:, 1]
    camera_x, camera_y, camera_z = image_camera.blinx_image_to_camera(xs, ys, z_values)
    points_xy_mm = np.column_stack((np.asarray(camera_x), np.asarray(camera_y))) * 1000.0

    edge_lengths_mm = []
    for index in range(4):
        start = points_xy_mm[index]
        end = points_xy_mm[(index + 1) % 4]
        edge_lengths_mm.append(float(np.linalg.norm(end - start)))

    long_side_mm = max(edge_lengths_mm)
    short_side_mm = min(edge_lengths_mm)
    center_x = float(np.mean(box_points[:, 0]))
    center_y = float(np.mean(box_points[:, 1]))
    center_cam_x, center_cam_y, center_cam_z = image_camera.blinx_image_to_camera(center_x, center_y, float(depth_mm))
    return {
        "long_side_mm": float(long_side_mm),
        "short_side_mm": float(short_side_mm),
        "camera_center_xyz": [
            float(center_cam_x),
            float(center_cam_y),
            float(center_cam_z),
        ],
    }


def score_depth_candidate(candidate: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, float]:
    expected_long = _coerce_float(cfg.get("depth_geom_expected_long_mm"), 100.0)
    expected_short = _coerce_float(cfg.get("depth_geom_expected_short_mm"), 50.0)
    expected_height = _coerce_float(cfg.get("depth_geom_expected_height_mm"), 70.0)
    long_tol = max(_coerce_float(cfg.get("depth_geom_long_tol_mm"), expected_long * 0.35), 1.0)
    short_tol = max(_coerce_float(cfg.get("depth_geom_short_tol_mm"), expected_short * 0.35), 1.0)
    height_tol = max(_coerce_float(cfg.get("depth_geom_height_tol_mm"), expected_height * 0.35), 1.0)
    planarity_max_std = max(_coerce_float(cfg.get("depth_geom_planarity_max_std_mm"), 3.0), 0.1)

    long_score = max(0.0, 1.0 - abs(float(candidate["long_side_mm"]) - expected_long) / long_tol)
    short_score = max(0.0, 1.0 - abs(float(candidate["short_side_mm"]) - expected_short) / short_tol)
    size_match_score = float(np.clip((long_score + short_score) * 0.5, 0.0, 1.0))
    planarity_score = float(
        np.clip(1.0 - float(candidate["height_std_mm"]) / planarity_max_std, 0.0, 1.0)
    )
    height_consistency_score = float(
        np.clip(
            1.0 - abs(float(candidate["median_height_mm"]) - expected_height) / height_tol,
            0.0,
            1.0,
        )
    )
    rectangularity_score = float(np.clip(candidate["rectangularity_score"], 0.0, 1.0))
    completeness_score = float(np.clip(candidate["completeness_score"], 0.0, 1.0))
    border_penalty = 0.2 if bool(candidate["touch_border"]) else 0.0
    geometry_score = float(
        np.clip(
            (
                0.35 * size_match_score
                + 0.20 * rectangularity_score
                + 0.20 * planarity_score
                + 0.15 * completeness_score
                + 0.10 * height_consistency_score
                - border_penalty
            ),
            0.0,
            1.0,
        )
    )
    return {
        "size_match_score": size_match_score,
        "rectangularity_score": rectangularity_score,
        "planarity_score": planarity_score,
        "completeness_score": completeness_score,
        "height_consistency_score": height_consistency_score,
        "geometry_score": geometry_score,
        "border_penalty": border_penalty,
    }


def build_depth_candidate(
    region: Dict[str, Any],
    rectangle_fit: Dict[str, Any],
    preprocess_result: Dict[str, Any],
    cfg: Dict[str, Any],
    image_camera: Any,
    image_shape: Sequence[int],
) -> Dict[str, Any]:
    x1, y1, _x2, _y2 = preprocess_result["workspace_roi"]
    global_mask = np.zeros(tuple(image_shape), dtype=bool)
    global_mask[y1 : y1 + region["mask"].shape[0], x1 : x1 + region["mask"].shape[1]] = region["mask"]

    local_box_points = np.asarray(rectangle_fit["box_points"], dtype=np.float32)
    global_box_points = local_box_points.copy()
    global_box_points[:, 0] += float(x1)
    global_box_points[:, 1] += float(y1)

    depth_filtered = preprocess_result["depth_filtered"]
    region_depth_values = depth_filtered[region["mask"] & np.isfinite(depth_filtered)]
    median_depth_mm = float(np.median(region_depth_values)) if region_depth_values.size else None
    mm_measure = (
        measure_rectangle_mm(global_box_points, median_depth_mm, image_camera)
        if median_depth_mm is not None
        else {"long_side_mm": 0.0, "short_side_mm": 0.0, "camera_center_xyz": [None, None, None]}
    )

    candidate = {
        "source": "depth_geom",
        "pixel_x": int(round(rectangle_fit["center_x"] + x1)),
        "pixel_y": int(round(rectangle_fit["center_y"] + y1)),
        "depth_mm": median_depth_mm,
        "angle_deg": float(rectangle_fit["angle_deg"]),
        "raw_pca_angle_deg": None,
        "score": 0.0,
        "mask": global_mask,
        "bbox": [
            int(np.floor(np.min(global_box_points[:, 0]))),
            int(np.floor(np.min(global_box_points[:, 1]))),
            int(np.ceil(np.max(global_box_points[:, 0]))),
            int(np.ceil(np.max(global_box_points[:, 1]))),
        ],
        "box_points": global_box_points,
        "long_side_px": float(rectangle_fit["long_side_px"]),
        "short_side_px": float(rectangle_fit["short_side_px"]),
        "long_side_mm": float(mm_measure["long_side_mm"]),
        "short_side_mm": float(mm_measure["short_side_mm"]),
        "median_height_mm": float(region["median_height_mm"]),
        "height_std_mm": float(region["height_std_mm"]),
        "valid_depth_count": int(region["valid_depth_count"]),
        "valid_depth_ratio": float(region["valid_depth_ratio"]),
        "touch_border": bool(region["touch_border"]),
        "rectangularity_score": float(rectangle_fit["rectangularity_score"]),
        "completeness_score": float(rectangle_fit["completeness_score"]),
        "camera_center_xyz": mm_measure["camera_center_xyz"],
        "axis_ratio": float(rectangle_fit["long_side_px"] / max(rectangle_fit["short_side_px"], 1e-6)),
        "angle_fallback": False,
    }
    candidate.update(score_depth_candidate(candidate, cfg))
    candidate["score"] = candidate["geometry_score"]
    candidate["is_valid"] = bool(
        median_depth_mm is not None
        and not candidate["touch_border"]
        and candidate["rectangularity_score"] >= _coerce_float(cfg.get("depth_geom_min_rectangularity"), 0.65)
        and candidate["completeness_score"] >= _coerce_float(cfg.get("depth_geom_min_completeness"), 0.60)
        and candidate["size_match_score"] > 0.0
    )
    return candidate


def rank_depth_candidates(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda item: (
            not bool(item["is_valid"]),
            bool(item["touch_border"]),
            -float(item["geometry_score"]),
            float(item["depth_mm"]) if item["depth_mm"] is not None else float("inf"),
        ),
    )


def render_depth_debug_panel(
    image: np.ndarray,
    depth_map: np.ndarray,
    preprocess_result: Dict[str, Any],
    height_map: np.ndarray,
    foreground_mask: np.ndarray,
    regions: Sequence[Dict[str, Any]],
    ranked_candidates: Sequence[Dict[str, Any]],
    selected_candidate: Optional[Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    rgb_image = np.ascontiguousarray(np.asarray(image).copy())
    depth_preview = _depth_to_preview(depth_map)
    height_preview = np.zeros_like(depth_preview)
    foreground_preview = np.zeros_like(depth_preview)
    regions_preview = np.zeros_like(depth_preview)
    rectangles_preview = rgb_image.copy()
    selected_overlay = depth_preview.copy()
    foreground_regions_preview = np.zeros_like(depth_preview)
    rectangles_selected_preview = rgb_image.copy()

    x1, y1, x2, y2 = preprocess_result["workspace_roi"]
    roi_height_map = np.asarray(height_map, dtype=np.float32)
    height_mask = np.isfinite(roi_height_map)
    if np.any(height_mask):
        clipped = np.clip(
            roi_height_map,
            0.0,
            max(_coerce_float(preprocess_result.get("board_depth_mm"), 1.0), _coerce_float(np.nanmax(roi_height_map), 1.0)),
        )
        normalized = np.zeros_like(clipped, dtype=np.uint8)
        valid_values = clipped[height_mask].astype(np.float32)
        min_value = float(np.min(valid_values))
        max_value = float(np.max(valid_values))
        if max_value - min_value <= 1e-6:
            normalized[height_mask] = 255
        else:
            scaled = ((valid_values - min_value) / (max_value - min_value) * 255.0).astype(np.uint8)
            normalized[height_mask] = scaled
        roi_height_rgb = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        height_preview[y1:y2, x1:x2] = cv2.cvtColor(roi_height_rgb, cv2.COLOR_BGR2RGB)

    foreground_preview[y1:y2, x1:x2] = _gray_to_rgb(foreground_mask.astype(np.uint8) * 255)
    foreground_regions_preview[y1:y2, x1:x2] = _gray_to_rgb(foreground_mask.astype(np.uint8) * 255)
    cv2.rectangle(height_preview, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 2)
    cv2.rectangle(foreground_preview, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 2)

    for index, region in enumerate(regions):
        color = (
            int((53 + index * 67) % 255),
            int((149 + index * 41) % 255),
            int((223 + index * 29) % 255),
        )
        region_rgb = np.zeros((region["mask"].shape[0], region["mask"].shape[1], 3), dtype=np.uint8)
        region_rgb[region["mask"]] = color
        target = regions_preview[y1 : y1 + region["mask"].shape[0], x1 : x1 + region["mask"].shape[1]]
        np.maximum(target, region_rgb, out=target)
        combined_target = foreground_regions_preview[y1 : y1 + region["mask"].shape[0], x1 : x1 + region["mask"].shape[1]]
        region_overlay = combined_target.copy()
        region_overlay[region["mask"]] = color
        combined_target[:] = cv2.addWeighted(region_overlay, 0.68, combined_target, 0.32, 0.0)
        _draw_mask_outline(combined_target, region["mask"], (255, 255, 255), thickness=2)

    cv2.rectangle(regions_preview, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 2)
    cv2.rectangle(foreground_regions_preview, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 2)
    _draw_mask_outline(foreground_regions_preview[y1:y2, x1:x2], foreground_mask, (255, 255, 255), thickness=2)

    for rank_index, candidate in enumerate(ranked_candidates):
        box_points = np.intp(np.round(candidate["box_points"]))
        box_color = (255, 165, 0) if rank_index else (0, 255, 0)
        cv2.drawContours(rectangles_preview, [box_points], 0, box_color, 2, cv2.LINE_AA)
        cv2.drawContours(rectangles_selected_preview, [box_points], 0, box_color, 2, cv2.LINE_AA)
        cv2.putText(
            rectangles_preview,
            f"#{rank_index + 1} S {candidate['geometry_score']:.2f}",
            (int(candidate["pixel_x"]) + 8, int(candidate["pixel_y"]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            box_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            rectangles_selected_preview,
            f"#{rank_index + 1} S {candidate['geometry_score']:.2f}",
            (int(candidate["pixel_x"]) + 8, int(candidate["pixel_y"]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            box_color,
            2,
            cv2.LINE_AA,
        )

    if selected_candidate is not None:
        mask = np.asarray(selected_candidate["mask"], dtype=bool)
        overlay = selected_overlay.copy()
        overlay[mask] = (0, 255, 0)
        selected_overlay = cv2.addWeighted(overlay, 0.28, selected_overlay, 0.72, 0.0)
        rectangles_selected_preview = _apply_mask_overlay(
            rectangles_selected_preview,
            mask,
            (0, 255, 0),
            alpha=0.26,
        )
        center = (int(selected_candidate["pixel_x"]), int(selected_candidate["pixel_y"]))
        box_points = np.intp(np.round(selected_candidate["box_points"]))
        cv2.drawContours(selected_overlay, [box_points], 0, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.drawContours(rectangles_selected_preview, [box_points], 0, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(selected_overlay, center, 6, (255, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(rectangles_selected_preview, center, 6, (255, 0, 0), -1, cv2.LINE_AA)
        _draw_angle_indicator(
            selected_overlay,
            center,
            float(selected_candidate["angle_deg"]),
            color=(255, 255, 255),
            length=int(max(70, min(selected_overlay.shape[0], selected_overlay.shape[1]) * 0.08)),
        )
        _draw_angle_indicator(
            rectangles_selected_preview,
            center,
            float(selected_candidate["angle_deg"]),
            color=(255, 255, 255),
            length=int(max(70, min(rectangles_selected_preview.shape[0], rectangles_selected_preview.shape[1]) * 0.08)),
        )
        _draw_multiline_label(
            selected_overlay,
            (center[0] + 24, center[1] - 150),
            [
                "Depth Selected",
                f"XY ({center[0]}, {center[1]})",
                f"Z {selected_candidate['depth_mm']:.1f} mm",
                f"Angle {selected_candidate['angle_deg']:.2f} deg",
                f"Geom {selected_candidate['geometry_score']:.2f}",
            ],
            font_scale=0.76,
            thickness=2,
        )
        _draw_multiline_label(
            rectangles_selected_preview,
            (center[0] + 24, center[1] - 150),
            [
                "Depth Selected",
                f"XY ({center[0]}, {center[1]})",
                f"Z {selected_candidate['depth_mm']:.1f} mm",
                f"Angle {selected_candidate['angle_deg']:.2f} deg",
                f"Geom {selected_candidate['geometry_score']:.2f}",
            ],
            font_scale=0.76,
            thickness=2,
        )
    cv2.rectangle(selected_overlay, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 2)
    cv2.rectangle(rectangles_selected_preview, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 2)

    analysis_panel = _stack_panel_rows(
        [
            [
                _build_panel_cell("Height Map", height_preview),
                _build_panel_cell("Foreground + Regions", foreground_regions_preview),
                _build_panel_cell("Rectangles + Selected", rectangles_selected_preview),
            ],
        ]
    )
    return {
        "height_map": height_preview,
        "foreground_mask": foreground_preview,
        "regions": regions_preview,
        "rectangles": rectangles_preview,
        "selected_overlay": selected_overlay,
        "analysis_panel": analysis_panel,
    }
