import cv2
import numpy as np
from ultralytics import YOLO
import math
import os

def calculate_angle(p1, p2):
    """
    计算两点连线与垂直方向的夹角（度数）
    p1: 上点 (shoulder_center)
    p2: 下点 (hip_center)
    """
    if p1 is None or p2 is None:
        return 0
    
    # 向量 p1 -> p2
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    if dy == 0:
        return 90.0
        
    # 计算与垂直方向（y轴向下）的夹角
    angle_rad = math.atan2(abs(dx), abs(dy))
    angle_deg = math.degrees(angle_rad)
    return angle_deg

import argparse

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Fall Detection on Video')
    parser.add_argument('--source', type=str, default=r"d:\华为选修课大作业\1272297698.mp4", help='Path to video file')
    args = parser.parse_args()

    # 视频文件路径
    video_path = args.source
    
    if not os.path.exists(video_path):
        print(f"错误: 找不到文件 {video_path}")
        return

    # 1. 加载 YOLOv8 Pose 模型
    print("正在加载模型...")
    try:
        model = YOLO('yolov8n-pose.pt')
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 2. 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    # 获取视频属性，用于保存结果（可选）
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"开始处理视频: {video_path}")
    print(f"分辨率: {width}x{height}, FPS: {fps}")
    print("按 'q' 退出...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频播放结束。")
            break

        # 3. 模型推理
        results = model(frame, stream=True, verbose=False)

        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints

            if boxes is None or keypoints is None:
                continue

            for i, box in enumerate(boxes):
                # 获取边界框
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = x2 - x1
                h = y2 - y1
                
                # 获取关键点
                kpts = keypoints.data[i].cpu().numpy()

                # 关键点索引
                # 5: Left Shoulder, 6: Right Shoulder
                # 11: Left Hip, 12: Right Hip
                conf_threshold = 0.5
                
                shoulder_l = kpts[5] if kpts[5][2] > conf_threshold else None
                shoulder_r = kpts[6] if kpts[6][2] > conf_threshold else None
                hip_l = kpts[11] if kpts[11][2] > conf_threshold else None
                hip_r = kpts[12] if kpts[12][2] > conf_threshold else None

                # 计算中心点
                shoulder_center = None
                if shoulder_l is not None and shoulder_r is not None:
                    shoulder_center = ((shoulder_l[0] + shoulder_r[0]) / 2, (shoulder_l[1] + shoulder_r[1]) / 2)
                
                hip_center = None
                if hip_l is not None and hip_r is not None:
                    hip_center = ((hip_l[0] + hip_r[0]) / 2, (hip_l[1] + hip_r[1]) / 2)

                # --- 摔倒概率计算 ---
                fall_prob = 0.0
                
                # 1. 宽高比
                aspect_ratio = w / h
                if aspect_ratio > 1.2:
                    fall_prob += 0.6
                elif aspect_ratio > 0.9:
                    fall_prob += 0.3
                
                # 2. 身体倾斜角度
                angle = 0
                if shoulder_center and hip_center:
                    angle = calculate_angle(shoulder_center, hip_center)
                    if angle > 60:
                        fall_prob += 0.4
                    elif angle > 45:
                        fall_prob += 0.2
                
                fall_prob = min(fall_prob, 1.0)

                # --- 绘制 ---
                color = (0, 255, 0)
                label = "Normal"
                
                if fall_prob > 0.5:
                    color = (0, 0, 255)
                    label = "Fall Detected"

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                text = f"{label}: {fall_prob:.2f}"
                cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                if shoulder_center and hip_center:
                    cv2.line(frame, (int(shoulder_center[0]), int(shoulder_center[1])), 
                             (int(hip_center[0]), int(hip_center[1])), (255, 0, 255), 2)

        # 显示画面
        # 调整窗口大小以适应屏幕（如果视频太大）
        display_frame = frame
        if width > 1280:
             display_frame = cv2.resize(frame, (1280, int(1280 * height / width)))
        
        cv2.imshow('Fall Detection - Video', display_frame)

        # 按 'q' 退出
        # waitKey 这里的参数可以控制播放速度，1ms 表示尽可能快，如果要按原速可以设为 int(1000/fps)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
