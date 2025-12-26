import cv2
import numpy as np
from ultralytics import YOLO
import math
import os
import time
import threading
from openai import OpenAI

# ================= 配置区域 =================
# ModelScope API 配置
API_KEY = "ms-f7c7041f-6807-4267-9236-dfc17a724ba0"
BASE_URL = "https://api-inference.modelscope.cn/v1/"

# 选择模型: 
# 1. "Qwen/Qwen2.5-72B-Instruct" (纯文本，速度快，稳定)
# 2. "Qwen/QVQ-72B-Preview" (视觉模型，可看图，但在免费API上可能较慢)
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct" 

# 报警冷却时间 (秒)，避免重复报警
ALERT_COOLDOWN = 30 
# ===========================================

# 初始化 AI 客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 全局变量控制报警状态
last_alert_time = 0
alert_lock = threading.Lock()

def send_notification(advice):
    """
    模拟发送通知给亲友或社区
    """
    print(f"\n{'!'*40}")
    print(f"【紧急通知】检测到老人跌倒！")
    print(f"【AI 建议处置方案】:\n{advice}")
    print(f"{'!'*40}\n")
    # 这里可以接入 钉钉/企业微信/短信 API
    # requests.post("webhook_url", data={...})

def get_ai_advice(fall_confidence):
    """
    调用 ModelScope 模型获取急救建议
    """
    print("正在请求 AI 急救建议...")
    try:
        prompt = f"系统检测到一位老人在家中跌倒，跌倒置信度为 {fall_confidence:.2f}。请立即给出3条简短、核心的急救处理建议，供监护人参考。"
        
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {'role': 'system', 'content': '你是一个专业的医疗急救助手。回答要简练、精准，适合紧急情况阅读。'},
                {'role': 'user', 'content': prompt}
            ],
            stream=False
        )
        advice = response.choices[0].message.content
        return advice
    except Exception as e:
        return f"AI 连接失败，请直接拨打 120。错误信息: {e}"

def process_alert(fall_prob):
    """
    处理报警的线程函数
    """
    global last_alert_time
    
    with alert_lock:
        current_time = time.time()
        if current_time - last_alert_time < ALERT_COOLDOWN:
            return # 冷却中，不报警

        last_alert_time = current_time
    
    # 1. 获取 AI 建议
    advice = get_ai_advice(fall_prob)
    
    # 2. 发送通知
    send_notification(advice)

def calculate_angle(p1, p2):
    """计算两点连线与垂直方向的夹角"""
    if p1 is None or p2 is None:
        return 0
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dy == 0:
        return 90.0
    angle_rad = math.atan2(abs(dx), abs(dy))
    return math.degrees(angle_rad)

def main():
    video_path = r"d:\华为选修课大作业\1272297698.mp4"
    if not os.path.exists(video_path):
        # 如果找不到绝对路径，尝试相对路径
        video_path = "1272297698.mp4"
    
    print(f"正在加载 YOLO 模型...")
    try:
        model = YOLO('yolov8n-pose.pt')
    except Exception as e:
        print(f"错误: 找不到 yolov8n-pose.pt 模型文件。请确保它在当前目录下。")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    print("系统启动成功！按 'q' 退出。")
    print("正在监测跌倒行为...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 推理
        results = model(frame, stream=True, verbose=False)

        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints

            if boxes is None or keypoints is None:
                continue

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = x2 - x1
                h = y2 - y1
                kpts = keypoints.data[i].cpu().numpy()

                # 简单跌倒检测逻辑 (复用原有逻辑)
                conf_threshold = 0.5
                shoulder_l = kpts[5] if kpts[5][2] > conf_threshold else None
                shoulder_r = kpts[6] if kpts[6][2] > conf_threshold else None
                hip_l = kpts[11] if kpts[11][2] > conf_threshold else None
                hip_r = kpts[12] if kpts[12][2] > conf_threshold else None

                shoulder_center = None
                if shoulder_l is not None and shoulder_r is not None:
                    shoulder_center = ((shoulder_l[0] + shoulder_r[0]) / 2, (shoulder_l[1] + shoulder_r[1]) / 2)
                
                hip_center = None
                if hip_l is not None and hip_r is not None:
                    hip_center = ((hip_l[0] + hip_r[0]) / 2, (hip_l[1] + hip_r[1]) / 2)

                fall_prob = 0.0
                aspect_ratio = w / h
                if aspect_ratio > 1.2: fall_prob += 0.6
                elif aspect_ratio > 0.9: fall_prob += 0.3
                
                if shoulder_center and hip_center:
                    angle = calculate_angle(shoulder_center, hip_center)
                    if angle > 60: fall_prob += 0.4
                    elif angle > 45: fall_prob += 0.2
                
                fall_prob = min(fall_prob, 1.0)

                # 绘制
                color = (0, 255, 0)
                if fall_prob > 0.5:
                    color = (0, 0, 255)
                    cv2.putText(frame, f"FALL DETECTED: {fall_prob:.2f}", (int(x1), int(y1)-30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # === 核心联动点 ===
                    # 如果检测到跌倒，且概率较高，触发 AI 报警
                    if fall_prob > 0.7:
                        # 开启新线程处理，避免阻塞视频播放
                        threading.Thread(target=process_alert, args=(fall_prob,)).start()

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # 显示
        display_frame = frame
        if frame.shape[1] > 1280:
             display_frame = cv2.resize(frame, (1280, int(1280 * frame.shape[0] / frame.shape[1])))
        cv2.imshow('AI Fall Detection & Alert System', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
