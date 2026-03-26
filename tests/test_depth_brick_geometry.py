import unittest

import cv2
import numpy as np

from ImageProcessing.depth_brick_geometry import (
    build_depth_candidate,
    build_height_map,
    estimate_board_depth_global_hist,
    extract_depth_foreground,
    fit_region_rectangle,
    measure_rectangle_mm,
    preprocess_depth_roi,
    rank_depth_candidates,
    render_depth_debug_panel,
    split_connected_regions,
)


class _FakeImageCamera:
    def blinx_image_to_camera(self, x, y, z):
        x = np.asarray(x, dtype=np.float32) / 1000.0
        y = np.asarray(y, dtype=np.float32) / 1000.0
        z = np.asarray(z, dtype=np.float32) / 1000.0
        return x, y, z


DEFAULT_CFG = {
    "primary_pick_depth_roi": [0, 0, 160, 120],
    "depth_min_valid_mm": 10.0,
    "depth_geom_board_estimation_mode": "global_hist",
    "depth_geom_hist_bin_mm": 2.0,
    "depth_geom_hist_peak_refine_window_mm": 3.0,
    "depth_geom_expected_long_mm": 60.0,
    "depth_geom_expected_short_mm": 40.0,
    "depth_geom_expected_height_mm": 70.0,
    "depth_geom_long_tol_mm": 15.0,
    "depth_geom_short_tol_mm": 12.0,
    "depth_geom_height_tol_mm": 10.0,
    "depth_geom_min_brick_height_mm": 30.0,
    "depth_geom_max_brick_height_mm": 100.0,
    "depth_geom_min_region_area_px": 400,
    "depth_geom_max_region_area_px": 50000,
    "depth_geom_median_kernel": 3,
    "depth_geom_open_kernel": 3,
    "depth_geom_close_kernel": 5,
    "depth_geom_border_margin_px": 8,
    "depth_geom_planarity_max_std_mm": 4.0,
    "depth_geom_min_rectangularity": 0.60,
    "depth_geom_min_completeness": 0.60,
}


def _draw_depth_brick(rect, shape=(120, 160), board_depth=1000.0, brick_depth=930.0):
    depth = np.full(shape, board_depth, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.uint8)
    box = cv2.boxPoints(rect)
    cv2.fillPoly(mask, [np.intp(np.round(box))], 1)
    depth[mask > 0] = brick_depth
    return depth, mask.astype(bool)


class BoardAndForegroundTests(unittest.TestCase):
    def test_estimate_board_depth_global_hist_ignores_small_outliers(self):
        values = np.concatenate(
            [
                np.full(2000, 1000.0, dtype=np.float32),
                np.full(30, 930.0, dtype=np.float32),
                np.array([0.0, -1.0, np.nan, np.inf], dtype=np.float32),
            ]
        )

        board_depth = estimate_board_depth_global_hist(values, DEFAULT_CFG)

        self.assertIsNotNone(board_depth)
        self.assertAlmostEqual(board_depth, 1000.0, delta=2.0)

    def test_foreground_pipeline_extracts_single_region(self):
        depth, _mask = _draw_depth_brick(((80, 60), (60, 40), 0))
        depth[0, 0] = np.nan
        depth[1, 1] = 0.0

        preprocess_result = preprocess_depth_roi(depth, DEFAULT_CFG)
        height_map = build_height_map(preprocess_result["board_depth_mm"], preprocess_result["depth_filtered"])
        foreground = extract_depth_foreground(height_map, preprocess_result["valid_mask"], DEFAULT_CFG)
        regions = split_connected_regions(foreground, height_map, preprocess_result["valid_mask"], DEFAULT_CFG)

        self.assertAlmostEqual(preprocess_result["board_depth_mm"], 1000.0, delta=2.0)
        self.assertEqual(len(regions), 1)
        self.assertGreater(regions[0]["area_px"], 1500)
        self.assertLess(regions[0]["height_std_mm"], 1.0)


class RectangleAndMeasurementTests(unittest.TestCase):
    def test_fit_region_rectangle_matches_rotated_brick(self):
        depth, _mask = _draw_depth_brick(((80, 60), (60, 40), 28))
        preprocess_result = preprocess_depth_roi(depth, DEFAULT_CFG)
        height_map = build_height_map(preprocess_result["board_depth_mm"], preprocess_result["depth_filtered"])
        foreground = extract_depth_foreground(height_map, preprocess_result["valid_mask"], DEFAULT_CFG)
        regions = split_connected_regions(foreground, height_map, preprocess_result["valid_mask"], DEFAULT_CFG)

        rectangle_fit = fit_region_rectangle(regions[0]["mask"], DEFAULT_CFG)

        self.assertIsNotNone(rectangle_fit)
        self.assertAlmostEqual(rectangle_fit["angle_deg"], 28.0, delta=6.0)
        self.assertGreater(rectangle_fit["rectangularity_score"], 0.7)
        self.assertGreater(rectangle_fit["completeness_score"], 0.7)

    def test_measure_rectangle_mm_uses_camera_projection(self):
        camera = _FakeImageCamera()
        box_points = np.array([[10, 10], [70, 10], [70, 50], [10, 50]], dtype=np.float32)

        measurement = measure_rectangle_mm(box_points, 1000.0, camera)

        self.assertAlmostEqual(measurement["long_side_mm"], 60.0, delta=0.5)
        self.assertAlmostEqual(measurement["short_side_mm"], 40.0, delta=0.5)


class CandidateScoringTests(unittest.TestCase):
    def test_build_depth_candidate_accepts_expected_brick(self):
        depth, _mask = _draw_depth_brick(((80, 60), (60, 40), 0))
        preprocess_result = preprocess_depth_roi(depth, DEFAULT_CFG)
        height_map = build_height_map(preprocess_result["board_depth_mm"], preprocess_result["depth_filtered"])
        foreground = extract_depth_foreground(height_map, preprocess_result["valid_mask"], DEFAULT_CFG)
        regions = split_connected_regions(foreground, height_map, preprocess_result["valid_mask"], DEFAULT_CFG)
        rectangle_fit = fit_region_rectangle(regions[0]["mask"], DEFAULT_CFG)

        candidate = build_depth_candidate(
            regions[0],
            rectangle_fit,
            preprocess_result,
            DEFAULT_CFG,
            _FakeImageCamera(),
            depth.shape,
        )

        self.assertTrue(candidate["is_valid"])
        self.assertAlmostEqual(candidate["long_side_mm"], 60.0, delta=3.0)
        self.assertAlmostEqual(candidate["short_side_mm"], 40.0, delta=3.0)
        self.assertGreater(candidate["geometry_score"], 0.7)

    def test_build_depth_candidate_rejects_touching_border_region(self):
        depth, _mask = _draw_depth_brick(((18, 60), (60, 40), 0))
        preprocess_result = preprocess_depth_roi(depth, DEFAULT_CFG)
        height_map = build_height_map(preprocess_result["board_depth_mm"], preprocess_result["depth_filtered"])
        foreground = extract_depth_foreground(height_map, preprocess_result["valid_mask"], DEFAULT_CFG)
        regions = split_connected_regions(foreground, height_map, preprocess_result["valid_mask"], DEFAULT_CFG)
        rectangle_fit = fit_region_rectangle(regions[0]["mask"], DEFAULT_CFG)

        candidate = build_depth_candidate(
            regions[0],
            rectangle_fit,
            preprocess_result,
            DEFAULT_CFG,
            _FakeImageCamera(),
            depth.shape,
        )

        self.assertTrue(candidate["touch_border"])
        self.assertFalse(candidate["is_valid"])

    def test_rank_depth_candidates_prefers_valid_non_border_higher_score(self):
        candidates = [
            {"is_valid": False, "touch_border": True, "geometry_score": 0.9, "depth_mm": 800.0},
            {"is_valid": True, "touch_border": False, "geometry_score": 0.8, "depth_mm": 900.0},
            {"is_valid": True, "touch_border": False, "geometry_score": 0.7, "depth_mm": 700.0},
        ]

        ranked = rank_depth_candidates(candidates)

        self.assertTrue(ranked[0]["is_valid"])
        self.assertEqual(ranked[0]["geometry_score"], 0.8)


class DebugPanelTests(unittest.TestCase):
    def test_render_depth_debug_panel_returns_expected_views(self):
        depth, _mask = _draw_depth_brick(((80, 60), (60, 40), 0))
        rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        preprocess_result = preprocess_depth_roi(depth, DEFAULT_CFG)
        height_map = build_height_map(preprocess_result["board_depth_mm"], preprocess_result["depth_filtered"])
        foreground = extract_depth_foreground(height_map, preprocess_result["valid_mask"], DEFAULT_CFG)
        regions = split_connected_regions(foreground, height_map, preprocess_result["valid_mask"], DEFAULT_CFG)
        rectangle_fit = fit_region_rectangle(regions[0]["mask"], DEFAULT_CFG)
        candidate = build_depth_candidate(
            regions[0],
            rectangle_fit,
            preprocess_result,
            DEFAULT_CFG,
            _FakeImageCamera(),
            depth.shape,
        )

        debug_images = render_depth_debug_panel(
            rgb,
            depth,
            preprocess_result,
            height_map,
            foreground,
            regions,
            [candidate],
            candidate,
        )

        self.assertEqual(
            set(debug_images.keys()),
            {"height_map", "foreground_mask", "regions", "rectangles", "selected_overlay", "analysis_panel"},
        )
        self.assertEqual(debug_images["analysis_panel"].ndim, 3)


if __name__ == "__main__":
    unittest.main()
