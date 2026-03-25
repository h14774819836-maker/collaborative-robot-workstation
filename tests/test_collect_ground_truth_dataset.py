import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from collect_ground_truth_dataset import (
    build_case_metadata,
    build_depth_preview,
    build_display_panel,
    get_next_case_id,
    write_case_bundle,
)


class _DummyPublicClass:
    cam_sn = "TEST-CAM-SN"


class _DummyImageCamera:
    camera_fx = 2275.5910981121006
    camera_fy = 2275.5910981121006
    camera_cx = 1351.0913172620878
    camera_cy = 884.5576714397653


class CaptureDatasetTests(unittest.TestCase):
    def setUp(self):
        self.rgb_image = np.full((48, 64, 3), 120, dtype=np.uint8)
        self.depth_map = np.linspace(500.0, 1500.0, 48 * 64, dtype=np.float32).reshape(48, 64)

    def test_get_next_case_id_skips_to_next_available_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir)
            (session_dir / "case_0001").mkdir()
            (session_dir / "case_0007").mkdir()
            (session_dir / "notes").mkdir()

            case_id = get_next_case_id(session_dir)

        self.assertEqual(case_id, "case_0008")

    def test_build_depth_preview_returns_rgb_preview_image(self):
        preview = build_depth_preview(self.depth_map)

        self.assertEqual(preview.shape, (48, 64, 3))
        self.assertEqual(preview.dtype, np.uint8)
        self.assertGreater(int(preview.max()), int(preview.min()))

    def test_build_display_panel_contains_header_and_combined_images(self):
        depth_preview = build_depth_preview(self.depth_map)

        panel = build_display_panel(
            self.rgb_image,
            depth_preview,
            case_id="case_0001",
            captured_at="2026-03-25T10:00:00+08:00",
            ground_truth_recorded_at="2026-03-25T10:01:00+08:00",
        )

        self.assertEqual(panel.ndim, 3)
        self.assertEqual(panel.shape[2], 3)
        self.assertGreater(panel.shape[0], self.rgb_image.shape[0])
        self.assertGreater(panel.shape[1], self.rgb_image.shape[1] * 2)

    def test_write_case_bundle_saves_expected_files_and_metadata(self):
        metadata = build_case_metadata(
            case_id="case_0001",
            session_name="session_demo",
            public_class=_DummyPublicClass(),
            image_camera=_DummyImageCamera(),
            rgb_image=self.rgb_image,
            depth_map=self.depth_map,
            captured_at="2026-03-25T10:00:00+08:00",
            ground_truth_recorded_at="2026-03-25T10:01:00+08:00",
            capture_tcp_pose=[1, 2, 3, 4, 5, 6],
            ground_truth_tcp_pose=[7, 8, 9, 10, 11, 12],
            capture_joint_pos=[13, 14, 15, 16, 17, 18],
            ground_truth_joint_pos=[19, 20, 21, 22, 23, 24],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir)
            case_dir = write_case_bundle(
                session_dir=session_dir,
                case_id="case_0001",
                rgb_image=self.rgb_image,
                depth_map=self.depth_map,
                metadata=metadata,
            )

            expected_files = {
                "rgb.npy",
                "depth.npy",
                "rgb_raw.png",
                "depth_preview.png",
                "display_rgb.png",
                "display_panel.png",
                "metadata.json",
            }
            self.assertEqual({path.name for path in case_dir.iterdir()}, expected_files)

            loaded_rgb = np.load(case_dir / "rgb.npy")
            loaded_depth = np.load(case_dir / "depth.npy")
            self.assertEqual(loaded_rgb.shape, self.rgb_image.shape)
            self.assertEqual(loaded_depth.shape, self.depth_map.shape)
            self.assertEqual(loaded_rgb.dtype, self.rgb_image.dtype)
            self.assertEqual(loaded_depth.dtype, self.depth_map.dtype)

            saved_metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_metadata["case_id"], "case_0001")
            self.assertEqual(saved_metadata["session_name"], "session_demo")
            self.assertEqual(saved_metadata["camera_serial_number"], "TEST-CAM-SN")
            self.assertEqual(saved_metadata["rgb_shape"], [48, 64, 3])
            self.assertEqual(saved_metadata["depth_shape"], [48, 64])

            for image_name in ("rgb_raw.png", "depth_preview.png", "display_rgb.png", "display_panel.png"):
                image = cv2.imread(str(case_dir / image_name), cv2.IMREAD_UNCHANGED)
                self.assertIsNotNone(image, msg=f"Failed to read {image_name}")


if __name__ == "__main__":
    unittest.main()
