# 摔倒检测项目 (Fall Detection using YOLOv8)

本项目使用 YOLOv8 Pose 模型进行实时摔倒检测。程序可以通过摄像头捕获视频流，或读取本地视频文件，分析人体姿态，并输出摔倒的概率。

## 环境准备

确保你已经安装了 Anaconda，并创建了虚拟环境（如 `fall_detection`）。

### 1. 依赖安装

如果你还没有安装依赖，请运行：

```bash
pip install -r requirements.txt
```

依赖列表：
- `ultralytics`: 用于 YOLOv8 模型
- `opencv-python`: 用于视频流处理和图像绘制
- `numpy`: 用于数值计算

## 运行项目

本项目包含两个主要脚本：

### 1. 实时摄像头检测 (`main.py`)

使用电脑摄像头进行实时检测：

```bash
python main.py
```

### 2. 视频文件检测 (`video_detect.py`)

检测本地视频文件中的摔倒情况。

**默认运行（检测 `1272297698.mp4`）：**
```bash
python video_detect.py
```

**指定其他视频文件：**
```bash
python video_detect.py --source "视频文件的路径.mp4"
```

例如：
```bash
python video_detect.py --source "d:\华为选修课大作业\407815802.mp4"
```

## 功能说明

程序会打开显示窗口，实时显示检测结果：
- **绿色框**: 正常状态 (Normal)
- **红色框**: 检测到摔倒 (Fall Detected)
- **概率值**: 显示在框上方，表示摔倒的可能性 (0.0 - 1.0)

### 检测逻辑
摔倒检测基于以下特征：
1. **宽高比 (Aspect Ratio)**: 摔倒的人通常宽度大于高度。
2. **身体倾斜角度**: 通过肩膀中心和臀部中心的连线计算身体相对于垂直方向的倾斜角。

## 退出
在播放过程中，按键盘上的 `q` 键可退出程序。
