import unittest

import cv2
import numpy as np

from ImageProcessing.image_recognition import (
    compute_pca_angle,
    extract_depth_stats,
    rank_candidates,
    resolve_shortest_rotation_delta,
)


DEFAULT_CFG = {
    "depth_min_valid_mm": 10.0,
    "depth_trim_percent": 5.0,
    "depth_min_valid_pixels": 20,
    "depth_min_valid_ratio": 0.2,
    "depth_erode_kernel": 3,
    "depth_erode_iterations": 1,
    "pca_min_axis_ratio": 1.10,
}


def build_rotated_mask(size, rect):
    mask = np.zeros(size, dtype=np.uint8)
    box = cv2.boxPoints(rect)
    cv2.fillPoly(mask, [np.intp(box)], 1)
    return mask.astype(bool)


class DepthStatsTests(unittest.TestCase):
    def test_extract_depth_stats_filters_invalid_and_outliers(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True

        depth_map = np.full((20, 20), 1000.0, dtype=np.float32)
        depth_map[5, 5] = 0.0
        depth_map[5, 6] = -1.0
        depth_map[5, 7] = np.nan
        depth_map[5, 8] = np.inf
        depth_map[5, 9] = 5000.0
        depth_map[5, 10] = 25.0

        stats = extract_depth_stats(mask, depth_map, DEFAULT_CFG)

        self.assertTrue(stats["is_valid"])
        self.assertAlmostEqual(stats["depth_mm"], 1000.0, delta=1.0)
        self.assertGreaterEqual(stats["valid_depth_count"], 60)
        self.assertGreater(stats["valid_depth_ratio"], 0.6)
        self.assertTrue(np.any(stats["inlier_mask"]))
        self.assertTrue(np.any(stats["rejected_mask"]))

    def test_extract_depth_stats_marks_small_valid_region_invalid(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:5, 3:5] = True
        depth_map = np.full((10, 10), 1000.0, dtype=np.float32)

        cfg = dict(DEFAULT_CFG)
        cfg["depth_min_valid_pixels"] = 10
        stats = extract_depth_stats(mask, depth_map, cfg)

        self.assertFalse(stats["is_valid"])
        self.assertEqual(stats["valid_depth_count"], 4)


class PcaAngleTests(unittest.TestCase):
    def test_compute_pca_angle_matches_rotated_rectangle(self):
        mask = build_rotated_mask((120, 120), ((60, 60), (64, 20), 30))

        angle_stats = compute_pca_angle(mask, DEFAULT_CFG)

        self.assertFalse(angle_stats["angle_fallback"])
        self.assertAlmostEqual(angle_stats["angle_deg"], 30.0, delta=5.0)

    def test_compute_pca_angle_falls_back_for_square(self):
        mask = build_rotated_mask((100, 100), ((50, 50), (40, 40), 15))

        angle_stats = compute_pca_angle(mask, DEFAULT_CFG)

        self.assertTrue(angle_stats["angle_fallback"])


class RankingAndRotationTests(unittest.TestCase):
    def test_rank_candidates_prefers_nearest_valid_target(self):
        candidates = [
            {"is_valid": True, "depth_mm": 900.0, "score": 0.7},
            {"is_valid": True, "depth_mm": 700.0, "score": 0.4},
            {"is_valid": False, "depth_mm": 500.0, "score": 0.99},
        ]

        ranked = rank_candidates(candidates)

        self.assertEqual(ranked[0]["depth_mm"], 700.0)
        self.assertTrue(ranked[0]["is_valid"])

    def test_resolve_shortest_rotation_delta_avoids_large_flip(self):
        resolved = resolve_shortest_rotation_delta(89.0, -89.0)
        self.assertAlmostEqual(resolved, 91.0, delta=0.01)

    def test_resolve_shortest_rotation_delta_uses_current_when_no_history(self):
        resolved = resolve_shortest_rotation_delta(None, -32.5)
        self.assertAlmostEqual(resolved, -32.5, delta=0.01)


if __name__ == "__main__":
    unittest.main()
