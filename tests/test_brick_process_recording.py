import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from brick_process_recording import BrickProcessRecorder, save_rgb_png


class BrickProcessRecorderTests(unittest.TestCase):
    def setUp(self):
        self.raw_rgb = np.full((40, 60, 3), 120, dtype=np.uint8)
        self.depth_map = np.linspace(500.0, 1500.0, 40 * 60, dtype=np.float32).reshape(40, 60)
        self.display_rgb = np.full((40, 60, 3), 200, dtype=np.uint8)

    def test_start_case_creates_primary_capture_bundle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = BrickProcessRecorder(base_dir=temp_dir, session_name="session_demo")
            case_id = recorder.start_case(
                process_num=0,
                process_node="3-2",
                raw_rgb=self.raw_rgb,
                depth_map=self.depth_map,
                display_rgb=self.display_rgb,
                public_snapshot={"tcp_pose": [1, 2, 3, 4, 5, 6]},
                extra={"command_target": [7, 8, 9, 10, 11, 12]},
            )

            case_dir = Path(temp_dir) / "session_demo" / case_id
            expected_files = {
                "primary_pick_rgb.npy",
                "primary_pick_depth.npy",
                "primary_pick_rgb_raw.png",
                "primary_pick_display_rgb.png",
                "primary_pick_depth_preview.png",
                "primary_pick_display_panel.png",
                "metadata.json",
            }
            self.assertEqual({path.name for path in case_dir.iterdir()}, expected_files)

            metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "in_progress")
            self.assertIn("primary_pick", metadata["captures"])
            self.assertEqual(metadata["captures"]["primary_pick"]["process_node"], "3-2")
            self.assertEqual(metadata["events"][0]["event_name"], "primary_pick_recorded")

    def test_record_event_and_secondary_capture_then_finalize(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = BrickProcessRecorder(base_dir=temp_dir, session_name="session_demo")
            recorder.start_case(
                process_num=1,
                process_node="3-2",
                raw_rgb=self.raw_rgb,
                depth_map=self.depth_map,
                display_rgb=self.display_rgb,
            )
            recorder.record_event(
                event_name="primary_rotation_command",
                process_node="3-3-1",
                public_snapshot={"joint_pos": [1, 2, 3, 4, 5, 6]},
                command_target=[10, 20, 30, 40, 50, 60],
                extra={"rotation_angle_deg": 12.5},
            )
            recorder.record_capture(
                capture_name="secondary_alignment",
                raw_rgb=self.raw_rgb,
                depth_map=self.depth_map,
                display_rgb=self.display_rgb,
                process_node="3-7-5",
                extra={"alignment_xy": [123.0, 456.0]},
            )
            recorder.finalize_case(
                status="cycle_complete",
                process_node="3-13",
                public_snapshot={"brick_process_num": 1},
                extra={"next_brick_process_num": 2},
            )

            case_dir = Path(temp_dir) / "session_demo" / "case_0001"
            metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "cycle_complete")
            self.assertEqual(metadata["final_process_node"], "3-13")
            self.assertIn("secondary_alignment", metadata["captures"])
            self.assertEqual(metadata["events"][1]["event_name"], "primary_rotation_command")
            self.assertEqual(metadata["events"][1]["extra"]["rotation_angle_deg"], 12.5)

            image = cv2.imread(str(case_dir / "secondary_alignment_display_panel.png"), cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(image)

    def test_save_rgb_png_supports_non_ascii_paths(self):
        rgb_image = np.zeros((10, 12, 3), dtype=np.uint8)
        rgb_image[:, :, 0] = 25
        rgb_image[:, :, 1] = 50
        rgb_image[:, :, 2] = 75

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "中文目录"
            output_dir.mkdir()
            image_path = output_dir / "测试图.png"

            save_rgb_png(image_path, rgb_image)

            image_buffer = np.fromfile(str(image_path), dtype=np.uint8)
            decoded = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
            self.assertIsNotNone(decoded)
            round_trip_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            self.assertTrue(np.array_equal(round_trip_rgb, rgb_image))


if __name__ == "__main__":
    unittest.main()
