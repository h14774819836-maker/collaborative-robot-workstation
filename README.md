# 协作机器人工作站

这是一个基于 `PyQt5` 的协作机器人上位机项目，用于整合视觉识别、3D 相机采集、JAKA 机器人控制和现场流程控制。

当前仓库为“纯源码版”：

- 不包含模型文件
- 不包含训练数据和采集数据
- 不包含现场实际配置
- 不包含厂商二进制运行库

## 主要能力

- 桌面操作界面与流程控制
- JAKA 机器人网络通信
- Mech-Eye 3D 相机采集
- 基于 ONNX 的目标检测与实例分割
- 标定参数与坐标转换

## 项目结构

- `Blinx_XXXY_Robot_Vision.py`
  主程序入口。
- `MainWindows.ui`
  Qt Designer 界面文件。
- `MainWindows.py`
  由 UI 文件生成的界面代码。
- `Config/`
  配置模板与配置目录。
- `ImageProcessing/`
  图像处理、推理代码和模型放置目录。
- `JakaPyLib/`
  JAKA 运行库放置目录。
- `Resources/`
  界面资源文件。
- `pic/`
  运行时输出目录。

## 运行环境

建议在 Windows 环境运行，并提前安装现场所需的设备 SDK。

Python 侧常用依赖：

- `PyQt5`
- `opencv-python`
- `numpy`
- `scipy`
- `onnxruntime` 或 `onnxruntime-gpu`
- `mecheye`

可以先安装通用依赖：

```bash
pip install PyQt5 opencv-python numpy scipy onnxruntime
```

如果现场使用 GPU 推理：

```bash
pip install PyQt5 opencv-python numpy scipy onnxruntime-gpu
```

## 启动方式

主程序入口：

```bash
python Blinx_XXXY_Robot_Vision.py
```

查看相机内参：

```bash
python get_camera_intrinsics.py
```

## 纯源码版说明

仓库中故意移除了以下内容：

- `ImageProcessing/*.onnx`
- `pic/` 下的大体积数据与采集结果
- `Config/config.ini`
- `JakaPyLib/` 下的二进制 SDK 文件

你需要按现场环境自行补齐这些内容。

### 模型文件

模型说明见 [ImageProcessing/MODELS.md](ImageProcessing/MODELS.md)。

当前代码直接引用的模型名是：

- `gangjin2-m.onnx`
- `hongzhuan-detect.onnx`
- `hongzhuan-seg2.onnx`

### 配置文件

请先复制：

- `Config/config.example.ini` -> `Config/config.ini`

然后根据现场实际设备修改：

- 相机序列号
- JAKA 控制器 IP 和端口
- PLC IP
- 视觉阈值
- 标定矩阵
- 工位点位

### JAKA 运行库

JAKA 相关说明见 [JakaPyLib/README.md](JakaPyLib/README.md)。

### 运行时图片目录

运行时输出目录说明见 [pic/README.md](pic/README.md)。

## 运行前检查

- 确认 `Config/config.ini` 已创建
- 确认模型文件已放入 `ImageProcessing/`
- 确认 JAKA 运行库已放入 `JakaPyLib/`
- 确认 Mech-Eye SDK 已安装
- 确认设备网络可达

## 后续建议

- 补充 `requirements.txt`
- 补充设备部署文档
- 补充标定流程文档
- 补充模型来源与版本说明
