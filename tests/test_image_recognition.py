import unittest

import cv2
import numpy as np

from ImageProcessing.image_recognition import (
    build_secondary_alignment_rgb_candidate,
    evaluate_depth_fallback_candidate,
    evaluate_rgb_candidate_quality,
    match_depth_candidate_to_rgb,
    compute_pca_angle,
    extract_depth_stats,
    rank_candidates,
    resolve_shortest_rotation_delta,
    select_secondary_alignment_candidate,
    select_primary_pick_candidate,
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

FUSION_CFG = {
    "primary_pick_rgb_depth_center_thresh_px": 80.0,
    "primary_pick_rgb_depth_angle_thresh_deg": 20.0,
    "primary_pick_rgb_depth_iou_thresh": 0.20,
    "primary_pick_rgb_depth_mm_thresh": 10.0,
    "primary_pick_rgb_low_score_thresh": 0.90,
    "primary_pick_rgb_low_valid_ratio_thresh": 0.70,
    "primary_pick_depth_fallback_geom_thresh": 0.88,
}


def build_rotated_mask(size, rect):
    mask = np.zeros(size, dtype=np.uint8)
    box = cv2.boxPoints(rect)
    cv2.fillPoly(mask, [np.intp(box)], 1)
    return mask.astype(bool)


def build_candidate(mask, *, pixel_x, pixel_y, angle_deg, depth_mm, score=0.95, valid_depth_ratio=0.9, source="rgb_seg", geometry_score=0.9, touch_border=False, is_valid=True, angle_fallback=False):
    ys, xs = np.where(mask)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    return {
        "source": source,
        "mask": mask,
        "bbox": bbox,
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "angle_deg": angle_deg,
        "depth_mm": depth_mm,
        "score": score,
        "geometry_score": geometry_score,
        "valid_depth_ratio": valid_depth_ratio,
        "touch_border": touch_border,
        "is_valid": is_valid,
        "angle_fallback": angle_fallback,
    }


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


class FusionDecisionTests(unittest.TestCase):
    def test_match_depth_candidate_to_rgb_finds_same_brick_not_depth_top1(self):
        rgb_mask = np.zeros((120, 120), dtype=bool)
        rgb_mask[50:90, 40:65] = True
        depth_match_mask = np.zeros((120, 120), dtype=bool)
        depth_match_mask[51:91, 41:66] = True
        depth_other_mask = np.zeros((120, 120), dtype=bool)
        depth_other_mask[10:50, 80:105] = True

        rgb_candidate = build_candidate(rgb_mask, pixel_x=52, pixel_y=70, angle_deg=-88.0, depth_mm=526.4, score=0.95)
        depth_match = build_candidate(
            depth_match_mask,
            pixel_x=53,
            pixel_y=71,
            angle_deg=-87.8,
            depth_mm=526.5,
            source="depth_geom",
            geometry_score=0.90,
        )
        depth_other = build_candidate(
            depth_other_mask,
            pixel_x=92,
            pixel_y=30,
            angle_deg=-88.1,
            depth_mm=526.3,
            source="depth_geom",
            geometry_score=0.98,
        )

        match_report = match_depth_candidate_to_rgb(rgb_candidate, [depth_other, depth_match], FUSION_CFG)

        self.assertTrue(match_report["match_found"])
        self.assertIs(match_report["matched_candidate"], depth_match)
        self.assertLess(match_report["metrics"]["center_distance_px"], 5.0)
        self.assertGreater(match_report["metrics"]["mask_iou"], 0.8)

    def test_select_primary_pick_candidate_keeps_rgb_when_same_brick_depth_matches(self):
        rgb_mask = np.zeros((120, 120), dtype=bool)
        rgb_mask[40:90, 45:70] = True
        depth_match_mask = np.zeros((120, 120), dtype=bool)
        depth_match_mask[41:91, 46:71] = True
        depth_top1_mask = np.zeros((120, 120), dtype=bool)
        depth_top1_mask[50:100, 80:105] = True

        rgb_candidate = build_candidate(rgb_mask, pixel_x=57, pixel_y=65, angle_deg=-89.0, depth_mm=526.5, score=0.96)
        depth_match = build_candidate(
            depth_match_mask,
            pixel_x=58,
            pixel_y=66,
            angle_deg=-88.7,
            depth_mm=526.6,
            source="depth_geom",
            geometry_score=0.89,
        )
        depth_top1 = build_candidate(
            depth_top1_mask,
            pixel_x=92,
            pixel_y=75,
            angle_deg=-89.1,
            depth_mm=526.4,
            source="depth_geom",
            geometry_score=0.97,
        )

        fusion_report = select_primary_pick_candidate(
            {"selected_candidate": rgb_candidate, "ranked_candidates": [rgb_candidate]},
            {"selected_candidate": depth_top1, "ranked_candidates": [depth_top1, depth_match]},
            FUSION_CFG,
        )

        self.assertEqual(fusion_report["decision_status"], "rgb_depth_agree_pass")
        self.assertEqual(fusion_report["selected_candidate"]["source"], "rgb_seg")
        self.assertTrue(fusion_report["selected_candidate"]["rgb_depth_match_found"])
        self.assertAlmostEqual(fusion_report["selected_candidate"]["match_mask_iou"], 0.859375, delta=0.05)

    def test_select_primary_pick_candidate_uses_depth_fallback_for_low_quality_rgb(self):
        rgb_mask = np.zeros((100, 100), dtype=bool)
        rgb_mask[10:40, 10:30] = True
        depth_mask = np.zeros((100, 100), dtype=bool)
        depth_mask[60:90, 60:85] = True

        rgb_candidate = build_candidate(
            rgb_mask,
            pixel_x=20,
            pixel_y=25,
            angle_deg=-45.0,
            depth_mm=520.0,
            score=0.55,
            valid_depth_ratio=0.35,
        )
        depth_candidate = build_candidate(
            depth_mask,
            pixel_x=72,
            pixel_y=75,
            angle_deg=-89.0,
            depth_mm=526.0,
            source="depth_geom",
            geometry_score=0.93,
        )

        fusion_report = select_primary_pick_candidate(
            {"selected_candidate": rgb_candidate, "ranked_candidates": [rgb_candidate]},
            {"selected_candidate": depth_candidate, "ranked_candidates": [depth_candidate]},
            FUSION_CFG,
        )

        self.assertEqual(fusion_report["decision_status"], "depth_fallback_pass")
        self.assertEqual(fusion_report["selected_candidate"]["source"], "depth_geom")

    def test_select_primary_pick_candidate_keeps_high_quality_rgb_without_depth_match(self):
        rgb_mask = np.zeros((100, 100), dtype=bool)
        rgb_mask[10:40, 10:30] = True
        depth_mask = np.zeros((100, 100), dtype=bool)
        depth_mask[60:90, 60:85] = True

        rgb_candidate = build_candidate(
            rgb_mask,
            pixel_x=20,
            pixel_y=25,
            angle_deg=-89.0,
            depth_mm=526.0,
            score=0.96,
            valid_depth_ratio=0.88,
        )
        depth_candidate = build_candidate(
            depth_mask,
            pixel_x=72,
            pixel_y=75,
            angle_deg=-89.0,
            depth_mm=526.0,
            source="depth_geom",
            geometry_score=0.95,
        )

        fusion_report = select_primary_pick_candidate(
            {"selected_candidate": rgb_candidate, "ranked_candidates": [rgb_candidate]},
            {"selected_candidate": depth_candidate, "ranked_candidates": [depth_candidate]},
            FUSION_CFG,
        )

        self.assertEqual(fusion_report["decision_status"], "rgb_only_pass")
        self.assertEqual(fusion_report["selected_candidate"]["source"], "rgb_seg")
        self.assertEqual(fusion_report["decision_warning"], "depth_mismatch")

    def test_build_secondary_alignment_rgb_candidate_handles_none(self):
        self.assertIsNone(build_secondary_alignment_rgb_candidate(None))

    def test_select_secondary_alignment_candidate_prefers_depth(self):
        depth_mask = np.zeros((100, 100), dtype=bool)
        depth_mask[20:60, 30:80] = True
        depth_candidate = build_candidate(
            depth_mask,
            pixel_x=55,
            pixel_y=40,
            angle_deg=0.0,
            depth_mm=241.7,
            source="depth_geom",
            geometry_score=0.91,
        )

        decision_report = select_secondary_alignment_candidate(
            {"selected_candidate": depth_candidate},
            [12.0, 18.0],
        )

        self.assertEqual(decision_report["decision_status"], "secondary_depth_pass")
        self.assertEqual(decision_report["selected_candidate"]["source"], "depth_geom")
        self.assertFalse(decision_report["rgb_fallback_used"])

    def test_select_secondary_alignment_candidate_falls_back_to_rgb(self):
        depth_mask = np.zeros((100, 100), dtype=bool)
        depth_mask[20:60, 30:80] = True
        invalid_depth_candidate = build_candidate(
            depth_mask,
            pixel_x=55,
            pixel_y=40,
            angle_deg=0.0,
            depth_mm=241.7,
            source="depth_geom",
            geometry_score=0.91,
            is_valid=False,
        )

        decision_report = select_secondary_alignment_candidate(
            {"selected_candidate": invalid_depth_candidate},
            [1226.5, 773.0],
        )

        self.assertEqual(decision_report["decision_status"], "secondary_rgb_fallback_pass")
        self.assertEqual(decision_report["selected_candidate"]["source"], "rgb_legacy")
        self.assertTrue(decision_report["rgb_fallback_used"])

    def test_select_secondary_alignment_candidate_returns_no_pick_when_both_missing(self):
        decision_report = select_secondary_alignment_candidate(
            {"selected_candidate": None},
            None,
        )

        self.assertEqual(decision_report["decision_status"], "secondary_no_pick")
        self.assertIsNone(decision_report["selected_candidate"])


if __name__ == "__main__":
    unittest.main()
