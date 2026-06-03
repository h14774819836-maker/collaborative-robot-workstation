# AGENTS.md

## Cursor Cloud specific instructions

### Product overview

This is a PyQt5 desktop application for controlling a collaborative robot (cobot) workstation. It integrates JAKA robot arm control, Mech-Eye 3D camera capture, and YOLOv8 ONNX-based object detection. The main entry point is `python Blinx_XXXY_Robot_Vision.py`.

### Running the application

- **Virtual display required**: The app is a PyQt5 GUI and needs an X display. Start Xvfb before launching: `Xvfb :99 -screen 0 1920x1080x24 &` then `export DISPLAY=:99`.
- **Config file**: `Config/config.ini` must exist. Copy from `Config/config.example.ini` if missing. The `[RelationalMatrix]` section values must be space-separated (not comma-separated) for the matrix parser in `strMatrix_to_Matrix()` to work correctly.
- **ONNX models**: Three `.onnx` files are needed in `ImageProcessing/`: `gangjin2-m.onnx`, `hongzhuan-detect.onnx`, `hongzhuan-seg2.onnx`. These are gitignored. For dev/testing without real models, generate dummy models (see below).
- **mecheye SDK**: The `mecheye` Python package is a vendor SDK not available on PyPI. A mock package is installed in the user site-packages for development. This mock returns empty data and no-ops for camera operations.
- **Hardware dependencies**: The app tries to connect to a JAKA robot (TCP socket) and Mech-Eye camera on a 5-second timer after startup. Without hardware, these connections will fail gracefully (robot connection logs "机器人连接失败", camera logs "Mech3D相机连接失败"). The app crashes ~6s after launch due to an unhandled `IndexError` in `ConnectCamera()` when no cameras are found. This is a known code issue — the for-loop in `ConnectCamera` doesn't properly guard against empty camera lists.

### Generating dummy ONNX models for dev

```python
python3 -c "
import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper

def make_detect(path, nc, h=640, w=640, na=8400):
    X = helper.make_tensor_value_info('images', TensorProto.FLOAT, [1,3,h,w])
    Y = helper.make_tensor_value_info('output0', TensorProto.FLOAT, [1,4+nc,na])
    c = helper.make_node('Constant', [], ['output0'],
        value=numpy_helper.from_array(np.zeros((1,4+nc,na),dtype=np.float32),'c'))
    m = helper.make_model(helper.make_graph([c],'g',[X],[Y]),opset_imports=[helper.make_opsetid('',13)])
    m.ir_version=8; onnx.save(m,path)

def make_seg(path, nc, nm=32, h=640, w=640, na=8400):
    X = helper.make_tensor_value_info('images', TensorProto.FLOAT, [1,3,h,w])
    Y0 = helper.make_tensor_value_info('output0', TensorProto.FLOAT, [1,4+nc+nm,na])
    Y1 = helper.make_tensor_value_info('output1', TensorProto.FLOAT, [1,nm,160,160])
    c0 = helper.make_node('Constant',[],['output0'],
        value=numpy_helper.from_array(np.zeros((1,4+nc+nm,na),dtype=np.float32),'c0'))
    c1 = helper.make_node('Constant',[],['output1'],
        value=numpy_helper.from_array(np.zeros((1,nm,160,160),dtype=np.float32),'c1'))
    m = helper.make_model(helper.make_graph([c0,c1],'g',[X],[Y0,Y1]),opset_imports=[helper.make_opsetid('',13)])
    m.ir_version=8; onnx.save(m,path)

make_detect('ImageProcessing/gangjin2-m.onnx', 1)
make_detect('ImageProcessing/hongzhuan-detect.onnx', 2)
make_seg('ImageProcessing/hongzhuan-seg2.onnx', 1)
"
```

### Lint & checks

- `flake8 --select=E9,F63,F7,F82 *.py ImageProcessing/*.py` — critical errors only
- `flake8 --max-line-length=200 *.py ImageProcessing/*.py` — full style check (many existing warnings)
- No automated tests exist in the codebase.

### Key caveats

- The codebase has no `__init__.py` files; all imports use flat module references from the project root. Always run from `/workspace`.
- `opencv-python-headless` is used instead of `opencv-python` to avoid conflicts with Qt's xcb plugin.
- The ONNX models produce "CUDAExecutionProvider not available" warnings when running on CPU — these are harmless.
- The app uses `QShortcut`, `QTimer`, and threaded TCP sockets — be aware of threading when debugging.
