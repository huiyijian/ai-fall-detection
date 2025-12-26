import cv2
import numpy as np
from ultralytics import YOLO
import math
import os
import time
import threading
import base64
from openai import OpenAI

# ================= 配置区域 =================
# ModelScope API 配置
API_KEY = "ms-f7c7041f-6807-4267-9236-dfc17a724ba0"
BASE_URL = "https://api-inference.modelscope.cn/v1/"

# 模型选择:
# 1. 快速反应模型 (Flash): 用于第一时间快速判断和给出核心指令
FLASH_MODEL_ID = "ZhipuAI/GLM-4v-Flash" 
# 2. 详细分析模型 (可选): 如果需要更深度的建议，后续可调用 Qwen2.5 或其他大模型
DETAIL_MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"

# 报警冷却时间 (秒)，避免重复报警
ALERT_COOLDOWN = 30 
# ===========================================

# 初始化 AI 客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 全局变量控制报警状态
last_alert_time = 0
alert_lock = threading.Lock()

def encode_image_to_base64(image):
    """将 OpenCV 图像转换为 Base64 字符串"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def send_notification(advice, model_name):
    """
    模拟发送通知给亲友或社区
    """
    print(f"\n{'!'*40}")
    print(f"【紧急通知】检测到老人跌倒！(Model: {model_name})")
    print(f"【AI 建议处置方案】:\n{advice}")
    print(f"{'!'*40}\n")

def get_ai_advice_flash(frame, fall_confidence):
    """
    调用 GLM-4v-Flash 视觉模型进行快速反应
    """
    print(f"正在请求 {FLASH_MODEL_ID} (视觉模型) 急救建议...")
    try:
        # 将当前帧转换为 base64
        base64_image = encode_image_to_base64(frame)
        
        prompt = f"检测到老人跌倒 (置信度 {fall_confidence:.2f})。请根据图片判断严重程度，并立即给出3条简短、核心的急救处理建议。"
        
        response = client.chat.completions.create(
            model=FLASH_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            stream=False
        )
        advice = response.choices[0].message.content
        return advice
    except Exception as e:
        print(f"Flash 模型调用失败: {e}")
        # 降级方案：使用纯文本模型
        return get_ai_advice_text_fallback(fall_confidence)

def get_ai_advice_text_fallback(fall_confidence):
    """
    降级方案：纯文本模型
    """
    print(f"降级使用 {DETAIL_MODEL_ID} (文本模型)...")
    try:
        prompt = f"系统检测到一位老人在家中跌倒，跌倒置信度为 {fall_confidence:.2f}。请立即给出3条简短、核心的急救处理建议。"
        response = client.chat.completions.create(
            model=DETAIL_MODEL_ID,
            messages=[
                {'role': 'system', 'content': '你是一个专业的医疗急救助手。'},
                {'role': 'user', 'content': prompt}
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"所有 AI 连接失败，请直接拨打 120。错误信息: {e}"

def process_alert(fall_prob, frame):
    """
    处理报警的线程函数
    """
    global last_alert_time
    
    with alert_lock:
        current_time = time.time()
        if current_time - last_alert_time < ALERT_COOLDOWN:
            return # 冷却中，不报警

        last_alert_time = current_time
    
    # 复制当前帧，避免多线程冲突
    frame_copy = frame.copy()
    
    # 1. 获取 AI 建议 (优先使用视觉模型 GLM-4v-Flash)
    advice = get_ai_advice_flash(frame_copy, fall_prob)
    
    # 2. 发送通知
    send_notification(advice, FLASH_MODEL_ID)

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
                        threading.Thread(target=process_alert, args=(fall_prob, frame)).start()

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
