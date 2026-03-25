import math
from typing import Dict, List, Optional

import cv2
import numpy as np

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

    def _run_rgbd_segmentation(self, image, mech_depth_map):
        boxes, segments, masks = self.model2(image, conf_threshold=self.conf, iou_threshold=self.iou)
        if len(boxes) == 0:
            rgb_debug, depth_debug = render_debug_overlays(
                image,
                mech_depth_map,
                None,
                self.rgbd_cfg["vision_debug"],
            )
            return rgb_debug, depth_debug, None

        candidates = []
        instances = self.model2.build_instance_records(boxes, segments, masks)
        for instance in instances:
            mask = instance["mask"].astype(bool)
            depth_stats = extract_depth_stats(mask, mech_depth_map, self.rgbd_cfg)
            angle_stats = compute_pca_angle(mask, self.rgbd_cfg)
            candidate = {
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
        debug_candidate = selected_candidate or (ranked_candidates[0] if ranked_candidates else None)
        rgb_debug, depth_debug = render_debug_overlays(
            image,
            mech_depth_map,
            debug_candidate,
            self.rgbd_cfg["vision_debug"],
        )
        return rgb_debug, depth_debug, selected_candidate

    def blinx_brick_image_rec(self, image, mech_depth_map):
        return self._run_rgbd_segmentation(image, mech_depth_map)

    def blinx_brick_image_rec2(self, image, mech_depth_map):
        return self._run_rgbd_segmentation(image, mech_depth_map)

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
