import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from offline_brick_recognition_review import (
    Tolerances,
    apply_recognizer_overrides,
    combine_primary_and_depth_analysis,
    build_primary_analysis_image,
    compute_primary_status,
    create_recognizer,
    discover_latest_session_dir,
    fit_image_to_screen,
    load_bgr_image,
    run_review,
)


def make_args(**overrides):
    defaults = {
        "session_dir": None,
        "case_id": None,
        "capture_scope": "both",
        "output_dir": None,
        "no_window": True,
        "conf": None,
        "iou": None,
        "depth_min_valid_mm": None,
        "depth_trim_percent": None,
        "depth_min_valid_pixels": None,
        "depth_min_valid_ratio": None,
        "depth_erode_kernel": None,
        "depth_erode_iterations": None,
        "pca_min_axis_ratio": None,
        "px_tol": 15.0,
        "angle_tol_deg": 5.0,
        "depth_tol_mm": 10.0,
        "robot_xy_tol_mm": 15.0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class _FakeRecognizer:
    def __init__(self):
        self.conf = 0.65
        self.iou = 0.45
        self.image_camera = mock.Mock()
        self.image_camera.blinx_image_to_camera.return_value = (0.1, 0.2, 0.784)
        self.rgbd_cfg = {
            "depth_min_valid_mm": 10.0,
            "depth_trim_percent": 5.0,
            "depth_min_valid_pixels": 200,
            "depth_min_valid_ratio": 0.2,
            "depth_erode_kernel": 3,
            "depth_erode_iterations": 1,
            "pca_min_axis_ratio": 1.1,
        }

    def _build_candidate(self, depth, x0, y0, x1, y1, score, angle_deg, is_valid=True):
        mask = np.zeros(depth.shape, dtype=bool)
        mask[y0:y1, x0:x1] = True
        return {
            "class_id": 0,
            "score": score,
            "bbox": [x0, y0, x1, y1],
            "mask": mask,
            "mask_foreground_count": int(mask.sum()),
            "pixel_x": int(round((x0 + x1) / 2.0)),
            "pixel_y": int(round((y0 + y1) / 2.0)),
            "depth_mm": 780.0,
            "angle_deg": angle_deg,
            "raw_pca_angle_deg": angle_deg,
            "valid_depth_count": int(mask.sum()),
            "valid_depth_ratio": 1.0,
            "inlier_mask": mask.copy(),
            "rejected_mask": np.zeros_like(mask, dtype=bool),
            "angle_fallback": False,
            "axis_ratio": 4.0,
            "is_valid": is_valid,
        }

    def blinx_brick_image_rec_debug(self, rgb, depth):
        candidate_1 = self._build_candidate(depth, 12, 8, 32, 20, 0.98, -89.0, is_valid=True)
        candidate_2 = self._build_candidate(depth, 34, 10, 44, 22, 0.61, -45.0, is_valid=True)
        return {
            "rgb_debug": rgb.copy(),
            "depth_debug": np.zeros_like(rgb),
            "candidates": [candidate_1, candidate_2],
            "ranked_candidates": [candidate_1, candidate_2],
            "selected_candidate": candidate_1,
            "selected_rank_index": 0,
        }

    def blinx_brick_image_rec2(self, rgb, depth):
        report = self.blinx_brick_image_rec_debug(rgb, depth)
        return report["rgb_debug"], report["depth_debug"], report["selected_candidate"]

    def blinx_brickandporcelain_image_rec(self, rgb):
        return rgb.copy(), [28.0, 18.0]

    def blinx_brick_depth_candidates(self, rgb, depth):
        candidate = self._build_candidate(depth, 12, 8, 32, 20, 0.88, -89.0, is_valid=True)
        candidate.update(
            {
                "source": "depth_geom",
                "geometry_score": 0.88,
                "long_side_mm": 120.0,
                "short_side_mm": 60.0,
                "median_height_mm": 70.0,
                "height_std_mm": 1.2,
                "size_match_score": 0.92,
                "planarity_score": 0.91,
                "rectangularity_score": 0.90,
                "completeness_score": 0.89,
                "height_consistency_score": 0.93,
                "touch_border": False,
                "box_points": np.array([[12, 8], [32, 8], [32, 20], [12, 20]], dtype=np.float32),
                "camera_center_xyz": [0.1, 0.2, 0.78],
            }
        )
        panel = np.full((80, 120, 3), 200, dtype=np.uint8)
        return {
            "rgb_debug": rgb.copy(),
            "depth_debug": panel.copy(),
            "debug_images": {
                "height_map": panel.copy(),
                "foreground_mask": panel.copy(),
                "regions": panel.copy(),
                "rectangles": panel.copy(),
                "selected_overlay": panel.copy(),
                "analysis_panel": panel.copy(),
            },
            "candidates": [candidate],
            "ranked_candidates": [candidate],
            "selected_candidate": candidate,
            "selected_rank_index": 0,
            "board_depth_mm": 840.0,
            "workspace_roi": [0, 0, 48, 32],
        }

    def blinx_brick_secondary_depth_candidates(self, rgb, depth):
        return self.blinx_brick_depth_candidates(rgb, depth)


class OfflineReviewTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((32, 48, 3), 120, dtype=np.uint8)
        self.depth = np.full((32, 48), 780.0, dtype=np.float32)

    def test_discover_latest_session_dir_picks_latest_named_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "brick_20260325_100001").mkdir()
            (base_dir / "brick_20260325_235959").mkdir()
            (base_dir / "brick_20260325_100001" / "case_0001").mkdir(parents=True)
            (base_dir / "brick_20260325_235959" / "case_0001").mkdir(parents=True)

            latest = discover_latest_session_dir(base_dir)

        self.assertEqual(latest.name, "brick_20260325_235959")

    def test_discover_latest_session_dir_skips_sessions_without_case_dirs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "20260325_171419" / "case_0001").mkdir(parents=True)
            (base_dir / "20260325_172951" / "offline_review").mkdir(parents=True)

            latest = discover_latest_session_dir(base_dir)

        self.assertEqual(latest.name, "20260325_171419")

    def test_discover_latest_session_dir_prefers_brick_process_records_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            brick_base = temp_path / "brick_process_records"
            ground_truth_base = temp_path / "ground_truth"
            (brick_base / "brick_20260325_174043" / "case_0001").mkdir(parents=True)
            (ground_truth_base / "20260326_100000" / "case_0001").mkdir(parents=True)

            with mock.patch("offline_brick_recognition_review.DEFAULT_RECORDS_DIR", brick_base):
                with mock.patch("offline_brick_recognition_review.GROUND_TRUTH_RECORDS_DIR", ground_truth_base):
                    latest = discover_latest_session_dir()

        self.assertEqual(latest.name, "brick_20260325_174043")

    def test_apply_recognizer_overrides_updates_thresholds_and_cfg(self):
        recognizer = _FakeRecognizer()
        args = make_args(
            conf=0.7,
            iou=0.33,
            depth_min_valid_mm=20.0,
            depth_trim_percent=7.0,
            depth_min_valid_pixels=300,
            depth_min_valid_ratio=0.45,
            depth_erode_kernel=5,
            depth_erode_iterations=2,
            pca_min_axis_ratio=1.4,
        )

        updated = apply_recognizer_overrides(recognizer, args)

        self.assertIs(updated, recognizer)
        self.assertEqual(updated.conf, 0.7)
        self.assertEqual(updated.iou, 0.33)
        self.assertEqual(updated.rgbd_cfg["depth_min_valid_mm"], 20.0)
        self.assertEqual(updated.rgbd_cfg["depth_trim_percent"], 7.0)
        self.assertEqual(updated.rgbd_cfg["depth_min_valid_pixels"], 300)
        self.assertEqual(updated.rgbd_cfg["depth_min_valid_ratio"], 0.45)
        self.assertEqual(updated.rgbd_cfg["depth_erode_kernel"], 5)
        self.assertEqual(updated.rgbd_cfg["depth_erode_iterations"], 2)
        self.assertEqual(updated.rgbd_cfg["pca_min_axis_ratio"], 1.4)

    def test_compute_primary_status_normalizes_angle_delta(self):
        offline_result = {"pixel_x": 100, "pixel_y": 200, "angle_deg": 89.0, "depth_mm": 500.0}
        runtime_reference = {"pixel_x": 100, "pixel_y": 200, "angle_deg": -89.0, "depth_mm": 500.0}

        status, delta = compute_primary_status(offline_result, runtime_reference, Tolerances(angle_deg=3.0))

        self.assertEqual(status, "runtime_consistent")
        self.assertAlmostEqual(delta["delta_angle_deg"], -2.0, delta=0.001)

    def test_build_primary_analysis_image_returns_panel_image(self):
        mask = np.zeros(self.depth.shape, dtype=bool)
        mask[8:20, 12:32] = True
        offline_result = {
            "mask": mask,
            "pixel_x": 22,
            "pixel_y": 14,
            "depth_mm": 780.0,
            "angle_deg": -89.0,
            "score": 0.98,
        }
        runtime_reference = {"pixel_x": 22, "pixel_y": 14, "depth_mm": 780.0, "angle_deg": -89.0}
        delta = {"delta_x_px": 0.0, "delta_y_px": 0.0, "delta_angle_deg": 0.0, "delta_depth_mm": 0.0}

        panel = build_primary_analysis_image(
            self.rgb,
            self.depth,
            offline_result,
            [offline_result],
            0,
            runtime_reference,
            "case_0001",
            "runtime_consistent",
            delta,
        )

        self.assertEqual(panel.ndim, 3)
        self.assertEqual(panel.shape[2], 3)
        self.assertGreater(panel.shape[0], self.rgb.shape[0])
        self.assertGreater(panel.shape[1], self.rgb.shape[1])
        self.assertNotEqual(int(panel.sum()), int(self.rgb.sum()))

    def test_combine_primary_and_depth_analysis_stacks_panels_vertically(self):
        primary_panel = np.full((60, 100, 3), 10, dtype=np.uint8)
        depth_panel = np.full((80, 120, 3), 20, dtype=np.uint8)

        combined = combine_primary_and_depth_analysis(primary_panel, depth_panel)

        self.assertEqual(combined.ndim, 3)
        self.assertEqual(combined.shape[2], 3)
        self.assertEqual(combined.shape[1], 120)
        self.assertEqual(combined.shape[0], 158)
        self.assertTrue(np.all(combined[0:60, 0:100] == 10))
        self.assertTrue(np.all(combined[78:, 0:120] == 20))

    def test_load_bgr_image_supports_non_ascii_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir) / "中文浏览"
            image_dir.mkdir()
            image_path = image_dir / "分析图.png"
            bgr_image = np.zeros((16, 20, 3), dtype=np.uint8)
            bgr_image[:, :, 0] = 10
            bgr_image[:, :, 1] = 20
            bgr_image[:, :, 2] = 30
            ok, encoded = cv2.imencode(".png", bgr_image)
            self.assertTrue(ok)
            encoded.tofile(str(image_path))

            loaded = load_bgr_image(image_path)

        self.assertTrue(np.array_equal(loaded, bgr_image))

    def test_fit_image_to_screen_scales_down_large_image(self):
        image = np.zeros((1800, 4800, 3), dtype=np.uint8)

        resized, scale = fit_image_to_screen(image, screen_size=(1920, 1080))

        self.assertLess(scale, 1.0)
        self.assertLessEqual(resized.shape[1], int(1920 * 0.92) + 1)
        self.assertLessEqual(resized.shape[0], int(1080 * 0.86) + 1)

    def test_fit_image_to_screen_keeps_small_image_size(self):
        image = np.zeros((600, 800, 3), dtype=np.uint8)

        resized, scale = fit_image_to_screen(image, screen_size=(1920, 1080))

        self.assertEqual(scale, 1.0)
        self.assertEqual(resized.shape, image.shape)

    @mock.patch("offline_brick_recognition_review.validate_model_files")
    @mock.patch("offline_brick_recognition_review.Blinx_image_rec")
    @mock.patch("offline_brick_recognition_review.Blinx_Public_Class")
    def test_create_recognizer_uses_public_class_and_applies_overrides(self, public_class_cls, recognizer_cls, _validate):
        recognizer = _FakeRecognizer()
        public_class = object()
        public_class_cls.return_value = public_class
        recognizer_cls.return_value = recognizer

        result = create_recognizer(make_args(conf=0.72, iou=0.21, depth_min_valid_pixels=333))

        public_class_cls.assert_called_once_with()
        recognizer_cls.assert_called_once_with(public_class)
        self.assertIs(result, recognizer)
        self.assertEqual(result.conf, 0.72)
        self.assertEqual(result.iou, 0.21)
        self.assertEqual(result.rgbd_cfg["depth_min_valid_pixels"], 333)

    def test_run_review_exports_case_outputs_and_summaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            session_dir = temp_path / "brick_20260325_174043"
            case_dir = session_dir / "case_0001"
            case_dir.mkdir(parents=True)

            np.save(case_dir / "primary_pick_rgb.npy", self.rgb)
            np.save(case_dir / "primary_pick_depth.npy", self.depth)
            np.save(case_dir / "secondary_alignment_rgb.npy", self.rgb)
            np.save(case_dir / "secondary_alignment_depth.npy", self.depth)

            metadata = {
                "case_id": "case_0001",
                "session_name": "brick_20260325_174043",
                "captures": {
                    "primary_pick": {
                        "extra": {
                            "pick_result": {
                                "pixel_x": 22,
                                "pixel_y": 14,
                                "depth_mm": 780.0,
                                "angle_deg": -89.0,
                                "score": 0.98,
                            }
                        }
                    },
                    "secondary_alignment": {
                        "extra": {
                            "alignment_result": [28.0, 18.0]
                        }
                    },
                },
            }
            (case_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

            output_dir = temp_path / "offline_output"
            args = make_args(session_dir=str(session_dir), output_dir=str(output_dir), no_window=True)

            fake_conversion = mock.Mock()
            fake_conversion.blinx_conversion.return_value = {"X": 100.0, "Y": 200.0}
            with mock.patch("offline_brick_recognition_review.create_recognizer", return_value=_FakeRecognizer()):
                with mock.patch("offline_brick_recognition_review.Blinx_3D_Conversion", return_value=fake_conversion):
                    result = run_review(args)

            self.assertEqual(result["output_root"], output_dir.resolve())
            self.assertEqual(len(result["summary_rows"]), 2)
            self.assertEqual(len(result["image_paths"]), 2)
            self.assertEqual(result["image_paths"][0].name, "primary_pick_analysis.png")
            self.assertEqual(result["image_paths"][1].name, "secondary_alignment_analysis.png")

            case_output_dir = output_dir / "case_0001"
            self.assertTrue((case_output_dir / "primary_pick_analysis.png").exists())
            self.assertTrue((case_output_dir / "secondary_alignment_analysis.png").exists())
            self.assertTrue((case_output_dir / "primary_pick_depth_height_map.png").exists())
            self.assertTrue((case_output_dir / "primary_pick_depth_foreground_mask.png").exists())
            self.assertTrue((case_output_dir / "primary_pick_depth_regions.png").exists())
            self.assertTrue((case_output_dir / "primary_pick_depth_rectangles.png").exists())
            self.assertTrue((case_output_dir / "primary_pick_depth_analysis.png").exists())
            self.assertTrue((case_output_dir / "primary_pick_depth_candidates.json").exists())
            self.assertTrue((case_output_dir / "secondary_alignment_depth_height_map.png").exists())
            self.assertTrue((case_output_dir / "secondary_alignment_depth_foreground_mask.png").exists())
            self.assertTrue((case_output_dir / "secondary_alignment_depth_regions.png").exists())
            self.assertTrue((case_output_dir / "secondary_alignment_depth_rectangles.png").exists())
            self.assertTrue((case_output_dir / "secondary_alignment_depth_analysis.png").exists())
            self.assertTrue((case_output_dir / "secondary_alignment_depth_candidates.json").exists())
            self.assertTrue((case_output_dir / "analysis.json").exists())
            self.assertTrue((output_dir / "summary.csv").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "review_queue.csv").exists())

            with (output_dir / "summary.csv").open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["status"] for row in rows}, {"runtime_consistent"})
            self.assertEqual(rows[0]["offline_candidate_count"], "2")
            self.assertEqual(rows[0]["offline_selected_rank"], "1")
            self.assertEqual(rows[0]["depth_selected_exists"], "True")
            self.assertEqual(rows[0]["depth_candidate_count"], "1")
            self.assertEqual(rows[0]["depth_status"], "runtime_consistent")
            self.assertEqual(rows[1]["capture_stage"], "secondary_alignment")
            self.assertEqual(rows[1]["depth_selected_exists"], "True")
            self.assertEqual(rows[1]["depth_candidate_count"], "1")
            self.assertEqual(rows[1]["depth_status"], "runtime_consistent")

            summary_json = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary_json["summary_count"], 2)
            self.assertEqual(summary_json["review_count"], 0)

            analysis_json = json.loads((case_output_dir / "analysis.json").read_text(encoding="utf-8"))
            primary_stage = analysis_json["stages"]["primary_pick"]
            self.assertEqual(primary_stage["offline_candidate_count"], 2)
            self.assertEqual(primary_stage["offline_selected_rank"], 1)
            self.assertEqual(len(primary_stage["offline_candidates"]), 2)
            self.assertTrue(primary_stage["offline_candidates"][0]["is_selected"])
            self.assertFalse(primary_stage["offline_candidates"][1]["is_selected"])
            self.assertIn("depth_geometry", primary_stage)
            self.assertEqual(primary_stage["depth_geometry"]["status"], "runtime_consistent")
            self.assertEqual(primary_stage["depth_geometry"]["board_depth_mm"], 840.0)
            self.assertEqual(len(primary_stage["depth_geometry"]["candidates"]), 1)
            secondary_stage = analysis_json["stages"]["secondary_alignment"]
            self.assertIn("depth_geometry", secondary_stage)
            self.assertEqual(secondary_stage["depth_geometry"]["status"], "runtime_consistent")
            self.assertEqual(secondary_stage["depth_geometry"]["board_depth_mm"], 840.0)
            self.assertEqual(len(secondary_stage["depth_geometry"]["candidates"]), 1)

    def test_run_review_prefers_ground_truth_metrics_when_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            session_dir = temp_path / "20260326_100000"
            case_dir = session_dir / "case_0001"
            case_dir.mkdir(parents=True)

            np.save(case_dir / "rgb.npy", self.rgb)
            np.save(case_dir / "depth.npy", self.depth)
            metadata = {
                "case_id": "case_0001",
                "session_name": "20260326_100000",
                "capture_tcp_pose": [100.0, 200.0, 950.0, 0.0, 0.0, 0.0],
                "ground_truth_tcp_pose": [100.0, 200.0, 336.0, 0.0, 0.0, 0.0],
                "capture_joint_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 10.0],
                "ground_truth_joint_pos": [0.0, 0.0, 0.0, 0.0, 0.0, -79.0],
            }
            (case_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            output_dir = temp_path / "offline_output"
            args = make_args(session_dir=str(session_dir), output_dir=str(output_dir), capture_scope="primary", no_window=True)

            fake_conversion = mock.Mock()
            fake_conversion.blinx_conversion.return_value = {"X": 100.0, "Y": 200.0}
            with mock.patch("offline_brick_recognition_review.create_recognizer", return_value=_FakeRecognizer()):
                with mock.patch("offline_brick_recognition_review.Blinx_3D_Conversion", return_value=fake_conversion):
                    result = run_review(args)

            self.assertEqual(len(result["summary_rows"]), 1)
            row = result["summary_rows"][0]
            self.assertEqual(row["depth_status"], "gt_consistent")
            self.assertAlmostEqual(float(row["depth_delta_pick_z_mm"]), 0.0, delta=0.01)

            analysis_json = json.loads((output_dir / "case_0001" / "analysis.json").read_text(encoding="utf-8"))
            depth_stage = analysis_json["stages"]["primary_pick"]["depth_geometry"]
            self.assertEqual(depth_stage["status"], "gt_consistent")
            self.assertAlmostEqual(float(depth_stage["manual_gt_metrics"]["delta_pick_z_mm"]), 0.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
