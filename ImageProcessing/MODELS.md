# Models

This repository is source-only.

The ONNX model files used by the project are intentionally not stored in Git.

Place the required model files in this directory before running the program. The current code references:

- `gangjin2-m.onnx`
- `hongzhuan-detect.onnx`
- `hongzhuan-seg2.onnx`

If you switch model names, update the paths in `ImageProcessing/image_recognition.py`.
