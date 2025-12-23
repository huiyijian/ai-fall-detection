import cv2
import numpy as np
from ultralytics import YOLO
import math

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
    # 垂直向量为 (0, 1)
    # 使用 atan2 计算角度，注意 y 轴向下
    angle_rad = math.atan2(abs(dx), abs(dy))
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def main():
    # 1. 加载 YOLOv8 Pose 模型
    # yolov8n-pose.pt 比较轻量，适合实时检测
    print("正在加载模型...")
    try:
        model = YOLO('yolov8n-pose.pt')
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确保网络连接正常，以便下载模型权重。")
        return

    # 2. 打开摄像头
    # 0 通常是默认摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("开始检测，按 'q' 退出...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法接收帧 (stream end?). Exiting ...")
            break

        # 3. 模型推理
        # stream=True 让推理更流畅
        results = model(frame, stream=True, verbose=False)

        for result in results:
            # 获取边界框和关键点
            boxes = result.boxes
            keypoints = result.keypoints

            if boxes is None or keypoints is None:
                continue

            # 遍历每一个检测到的人
            for i, box in enumerate(boxes):
                # 获取边界框坐标 (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = x2 - x1
                h = y2 - y1
                
                # 获取该人的关键点
                # data[0] 是 (num_points, 3) -> (x, y, conf)
                kpts = keypoints.data[i].cpu().numpy()

                # 提取关键点索引 (COCO format)
                # 5: Left Shoulder, 6: Right Shoulder
                # 11: Left Hip, 12: Right Hip
                
                # 检查关键点置信度，如果太低则不计算
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

                # --- 摔倒概率计算逻辑 ---
                fall_prob = 0.0
                
                # 因素 1: 宽高比 (Aspect Ratio)
                # 正常站立 w < h, ratio < 1
                # 摔倒 w > h, ratio > 1
                aspect_ratio = w / h
                
                # 简单的线性映射，如果 ratio > 1.2, 概率很高
                # Sigmoid-like 逻辑或分段线性
                if aspect_ratio > 1.2:
                    fall_prob += 0.6
                elif aspect_ratio > 0.9:
                    fall_prob += 0.3
                
                # 因素 2: 身体倾斜角度
                angle = 0
                if shoulder_center and hip_center:
                    angle = calculate_angle(shoulder_center, hip_center)
                    # 如果角度大于 45 度，增加摔倒概率
                    if angle > 60:
                        fall_prob += 0.4
                    elif angle > 45:
                        fall_prob += 0.2
                
                # 限制概率在 0.0 到 1.0 之间
                fall_prob = min(fall_prob, 1.0)

                # --- 绘制结果 ---
                # 颜色：绿色为安全，红色为摔倒
                color = (0, 255, 0)
                label = "Normal"
                
                if fall_prob > 0.5:
                    color = (0, 0, 255)
                    label = "Fall Detected"

                # 画框
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # 写文字 (概率)
                text = f"{label}: {fall_prob:.2f}"
                cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # 画骨架连线 (可选，为了视觉效果)
                if shoulder_center:
                    cv2.circle(frame, (int(shoulder_center[0]), int(shoulder_center[1])), 5, (255, 255, 0), -1)
                if hip_center:
                    cv2.circle(frame, (int(hip_center[0]), int(hip_center[1])), 5, (255, 255, 0), -1)
                if shoulder_center and hip_center:
                    cv2.line(frame, (int(shoulder_center[0]), int(shoulder_center[1])), 
                             (int(hip_center[0]), int(hip_center[1])), (255, 0, 255), 2)

        # 显示画面
        cv2.imshow('Fall Detection', frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
