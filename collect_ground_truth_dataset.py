from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from Blinx_Public_Class import Blinx_Public_Class
from ImageProcessing.BLX_Image_Camera import Blinx_Image_Camera


CASE_ID_PATTERN = re.compile(r"^case_(\d{4})$")


class CaptureSessionQuit(Exception):
    pass


class RepeatCurrentCase(Exception):
    pass


class SkipCurrentCase(Exception):
    pass


@dataclass
class RuntimeContext:
    public_class: Blinx_Public_Class
    image_camera: Blinx_Image_Camera
    mech_camera: Any
    jaka_socket: Any


@dataclass
class FrozenCapture:
    rgb_image: np.ndarray
    depth_map: np.ndarray
    captured_at: str
    capture_tcp_pose: list[float]
    capture_joint_pos: list[float]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture lab RGB/depth data and matching robot ground-truth poses.",
    )
    parser.add_argument(
        "--session-name",
        default=None,
        help="Optional session directory name. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to pic/cam1/ground_truth/<session-name>/",
    )
    return parser.parse_args(argv)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def build_session_name(session_name: Optional[str]) -> str:
    if session_name:
        return session_name.strip()
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_output_dir(output_dir: Optional[str], session_name: str) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    return (Path("pic") / "cam1" / "ground_truth" / session_name).resolve()


def normalize_pose(values: Any) -> Optional[list[float]]:
    if not isinstance(values, (list, tuple)) or len(values) < 6:
        return None
    try:
        return [float(values[index]) for index in range(6)]
    except (TypeError, ValueError):
        return None


def get_next_case_id(session_dir: Path) -> str:
    max_index = 0
    if session_dir.exists():
        for entry in session_dir.iterdir():
            if not entry.is_dir():
                continue
            match = CASE_ID_PATTERN.match(entry.name)
            if match:
                max_index = max(max_index, int(match.group(1)))
    return f"case_{max_index + 1:04d}"


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
    case_id: str,
    captured_at: str,
    ground_truth_recorded_at: str,
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
        f"Captured At: {captured_at}",
        f"Ground Truth Recorded At: {ground_truth_recorded_at}",
    ]
    for index, line in enumerate(text_lines):
        origin = (16, 28 + index * 24)
        cv2.putText(panel, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    return panel


def save_rgb_png(path: Path, rgb_image: np.ndarray) -> None:
    bgr_image = cv2.cvtColor(np.ascontiguousarray(rgb_image), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr_image):
        raise IOError(f"Failed to save image: {path}")


def build_case_metadata(
    *,
    case_id: str,
    session_name: str,
    public_class: Blinx_Public_Class,
    image_camera: Blinx_Image_Camera,
    rgb_image: np.ndarray,
    depth_map: np.ndarray,
    captured_at: str,
    ground_truth_recorded_at: str,
    capture_tcp_pose: Sequence[float],
    ground_truth_tcp_pose: Sequence[float],
    capture_joint_pos: Sequence[float],
    ground_truth_joint_pos: Sequence[float],
) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "session_name": session_name,
        "captured_at": captured_at,
        "ground_truth_recorded_at": ground_truth_recorded_at,
        "camera_serial_number": public_class.cam_sn,
        "rgb_shape": list(rgb_image.shape),
        "rgb_dtype": str(rgb_image.dtype),
        "depth_shape": list(depth_map.shape),
        "depth_dtype": str(depth_map.dtype),
        "capture_tcp_pose": list(capture_tcp_pose),
        "ground_truth_tcp_pose": list(ground_truth_tcp_pose),
        "capture_joint_pos": list(capture_joint_pos),
        "ground_truth_joint_pos": list(ground_truth_joint_pos),
        "pose_units": {"xyz": "mm", "rxyz": "deg"},
        "algorithm_camera_intrinsics": {
            "fx": float(image_camera.camera_fx),
            "fy": float(image_camera.camera_fy),
            "cx": float(image_camera.camera_cx),
            "cy": float(image_camera.camera_cy),
        },
        "array_layout": {
            "rgb": "HWC",
            "depth": "HW",
        },
    }


def write_case_bundle(
    *,
    session_dir: Path,
    case_id: str,
    rgb_image: np.ndarray,
    depth_map: np.ndarray,
    metadata: Dict[str, Any],
) -> Path:
    display_rgb = build_display_rgb(rgb_image)
    depth_preview = build_depth_preview(depth_map)
    display_panel = build_display_panel(
        display_rgb,
        depth_preview,
        case_id=case_id,
        captured_at=str(metadata["captured_at"]),
        ground_truth_recorded_at=str(metadata["ground_truth_recorded_at"]),
    )

    temp_case_dir = session_dir / f".{case_id}_tmp"
    case_dir = session_dir / case_id
    if temp_case_dir.exists():
        shutil.rmtree(temp_case_dir)
    temp_case_dir.mkdir(parents=True, exist_ok=False)

    try:
        np.save(temp_case_dir / "rgb.npy", np.asarray(rgb_image))
        np.save(temp_case_dir / "depth.npy", np.asarray(depth_map))
        save_rgb_png(temp_case_dir / "rgb_raw.png", np.asarray(rgb_image))
        save_rgb_png(temp_case_dir / "display_rgb.png", display_rgb)
        save_rgb_png(temp_case_dir / "depth_preview.png", depth_preview)
        save_rgb_png(temp_case_dir / "display_panel.png", display_panel)
        (temp_case_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_case_dir.replace(case_dir)
    except Exception:
        shutil.rmtree(temp_case_dir, ignore_errors=True)
        raise

    return case_dir


def load_runtime() -> RuntimeContext:
    config_path = Path("Config") / "config.ini"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    from Blinx_Jaka_Socket import Blinx_Jaka_Socket
    from Blinx_Mech3D_Viewer_SDK import Blinx_Mech3D_Viewer_SDK

    public_class = Blinx_Public_Class()
    image_camera = Blinx_Image_Camera()
    mech_camera = Blinx_Mech3D_Viewer_SDK(public_class)
    mech_camera.ConnectCamera()
    if not public_class.mech_connected:
        raise RuntimeError("Mech3D camera connection failed.")

    jaka_socket = Blinx_Jaka_Socket(public_class)
    if not jaka_socket.jaka_connect_sucess:
        try:
            mech_camera.DisConnectCamera()
        except Exception:
            pass
        raise RuntimeError("JAKA TCP socket connection failed.")

    return RuntimeContext(
        public_class=public_class,
        image_camera=image_camera,
        mech_camera=mech_camera,
        jaka_socket=jaka_socket,
    )


def shutdown_runtime(runtime: Optional[RuntimeContext]) -> None:
    if runtime is None:
        return

    try:
        runtime.jaka_socket.blinx_jaka_close()
    except Exception:
        pass

    try:
        if runtime.public_class.mech_connected:
            runtime.mech_camera.DisConnectCamera()
    except Exception:
        pass


def wait_for_robot_pose(runtime: RuntimeContext, timeout_s: float = 5.0, poll_interval: float = 0.2) -> Tuple[list[float], list[float]]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        runtime.jaka_socket.blinx_get_joint_pos()
        runtime.jaka_socket.blinx_get_tcp_pos()
        time.sleep(poll_interval)
        joint_pos = normalize_pose(runtime.public_class.joint_pos)
        tcp_pos = normalize_pose(runtime.public_class.tcp_pos)
        if joint_pos is not None and tcp_pos is not None:
            return joint_pos, tcp_pos
    raise TimeoutError("Timed out while waiting for robot joint/tcp pose data.")


def capture_current_frame(runtime: RuntimeContext) -> FrozenCapture:
    rgb_image, depth_map, _point_cloud = runtime.mech_camera.GrabImages()
    if rgb_image is None or depth_map is None:
        raise RuntimeError("Camera returned empty RGB/depth data.")

    rgb_image = np.ascontiguousarray(np.asarray(rgb_image))
    depth_map = np.ascontiguousarray(np.asarray(depth_map))
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"Unexpected RGB image shape: {rgb_image.shape}")
    if depth_map.ndim != 2:
        raise ValueError(f"Unexpected depth map shape: {depth_map.shape}")

    capture_joint_pos, capture_tcp_pose = wait_for_robot_pose(runtime)
    return FrozenCapture(
        rgb_image=rgb_image,
        depth_map=depth_map,
        captured_at=now_iso(),
        capture_tcp_pose=capture_tcp_pose,
        capture_joint_pos=capture_joint_pos,
    )


def _prompt_command(message: str) -> str:
    return input(message).strip().lower()


def prompt_for_capture() -> None:
    command = _prompt_command(
        "\n按 Enter 抓取当前帧，输入 q 退出，输入 s 跳过本组：",
    )
    if command == "":
        return
    if command == "q":
        raise CaptureSessionQuit()
    if command == "s":
        raise SkipCurrentCase()
    if command == "r":
        raise RepeatCurrentCase()
    print(f"未识别命令: {command!r}，请重新输入。")
    prompt_for_capture()


def prompt_for_ground_truth() -> None:
    command = _prompt_command(
        "\n将机械臂手动移动到理想抓取位后按 Enter 记录真值，输入 r 重拍本组，输入 s 取消本组，输入 q 退出：",
    )
    if command == "":
        return
    if command == "q":
        raise CaptureSessionQuit()
    if command == "r":
        raise RepeatCurrentCase()
    if command == "s":
        raise SkipCurrentCase()
    print(f"未识别命令: {command!r}，请重新输入。")
    prompt_for_ground_truth()


def print_session_header(runtime: RuntimeContext, session_name: str, output_dir: Path) -> None:
    camera_model = "unknown"
    if getattr(runtime.mech_camera, "cam_index", -1) >= 0:
        try:
            camera_model = runtime.mech_camera.camera_infos[runtime.mech_camera.cam_index].model
        except Exception:
            camera_model = "unknown"

    print("=" * 72)
    print("实验室黄金数据采集器")
    print(f"Session Name : {session_name}")
    print(f"Output Dir   : {output_dir}")
    print(f"Camera SN    : {runtime.public_class.cam_sn or 'unknown'}")
    print(f"Camera Model : {camera_model}")
    print("Robot Socket : connected")
    print("Array Layout : RGB=(H, W, 3), Depth=(H, W)")
    print("=" * 72)


def run_capture_session(session_name: str, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime: Optional[RuntimeContext] = None
    saved_cases = 0

    try:
        runtime = load_runtime()
        wait_for_robot_pose(runtime, timeout_s=8.0)
        print_session_header(runtime, session_name, output_dir)

        while True:
            case_id = get_next_case_id(output_dir)
            print(f"\n准备采集 {case_id}")
            try:
                prompt_for_capture()
                frozen_capture = capture_current_frame(runtime)
                print(
                    f"已冻结 {case_id} 画面: rgb_shape={tuple(frozen_capture.rgb_image.shape)}, "
                    f"depth_shape={tuple(frozen_capture.depth_map.shape)}",
                )
                print(f"拍照瞬间 TCP   : {frozen_capture.capture_tcp_pose}")
                print(f"拍照瞬间关节姿态: {frozen_capture.capture_joint_pos}")

                prompt_for_ground_truth()
                ground_truth_joint_pos, ground_truth_tcp_pose = wait_for_robot_pose(runtime)
                ground_truth_recorded_at = now_iso()

                metadata = build_case_metadata(
                    case_id=case_id,
                    session_name=session_name,
                    public_class=runtime.public_class,
                    image_camera=runtime.image_camera,
                    rgb_image=frozen_capture.rgb_image,
                    depth_map=frozen_capture.depth_map,
                    captured_at=frozen_capture.captured_at,
                    ground_truth_recorded_at=ground_truth_recorded_at,
                    capture_tcp_pose=frozen_capture.capture_tcp_pose,
                    ground_truth_tcp_pose=ground_truth_tcp_pose,
                    capture_joint_pos=frozen_capture.capture_joint_pos,
                    ground_truth_joint_pos=ground_truth_joint_pos,
                )
                case_dir = write_case_bundle(
                    session_dir=output_dir,
                    case_id=case_id,
                    rgb_image=frozen_capture.rgb_image,
                    depth_map=frozen_capture.depth_map,
                    metadata=metadata,
                )
                saved_cases += 1
                print(f"已保存 {case_id} -> {case_dir}")
                print(f"真值 TCP      : {ground_truth_tcp_pose}")
                print(f"真值关节姿态   : {ground_truth_joint_pos}")
            except RepeatCurrentCase:
                print(f"{case_id} 已重置，请重新采集。")
                continue
            except SkipCurrentCase:
                print(f"{case_id} 已取消，不保存任何文件。")
                continue
    except CaptureSessionQuit:
        print("\n收到退出指令，正在结束采集。")
    finally:
        shutdown_runtime(runtime)

    print(f"\n本次会话共保存 {saved_cases} 组数据。")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    session_name = build_session_name(args.session_name)
    output_dir = resolve_output_dir(args.output_dir, session_name)
    try:
        return run_capture_session(session_name, output_dir)
    except KeyboardInterrupt:
        print("\n用户中断，采集结束。")
        return 130
    except Exception as exc:
        print(f"\n采集失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
