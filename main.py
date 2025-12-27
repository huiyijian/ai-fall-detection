import cv2
import numpy as np
import time
import threading
import json
from pathlib import Path
from vlm_analyzer import VLMAnalyzer


class VideoStreamSimulator:
    """
    视频流模拟器
    模拟实时摄像头行为，支持键盘控制中断，不等待视频播放结束
    """
    def __init__(self, video_path, loop=True):
        self.video_path = Path(video_path) if not isinstance(video_path, int) else None
        self.loop = loop
        self.cap = None
        self.fps = 30
        self.frame_count = 0
        self.paused = False
        self.is_camera = isinstance(video_path, int)
        
    def open(self):
        if self.is_camera:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开摄像头: {self.video_path}")
            self.fps = 30
            print(f"已打开摄像头: {self.video_path}")
            return True
            
        if not self.video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {self.video_path}")
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"视频加载成功: {self.video_path.name}")
        print(f"分辨率: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"FPS: {self.fps}, 总帧数: {self.frame_count}")
        return True
        
    def read_frame(self):
        if self.cap is None or not self.cap.isOpened() or self.paused:
            return None, False
            
        ret, frame = self.cap.read()
        
        if not ret and self.loop and not self.is_camera:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            
        return frame, ret
        
    def toggle_pause(self):
        if self.is_camera:
            return False
        self.paused = not self.paused
        return self.paused
        
    def release(self):
        if self.cap is not None:
            self.cap.release()
            print("视频流已释放")


class FallDetector:
    """
    跌倒检测器 - 基于 YOLOv8-Pose
    """
    def __init__(self, model_path='yolov8n-pose.pt'):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        print("YOLOv8-Pose 模型加载完成")
        
    def detect(self, frame):
        """
        检测人体姿态
        返回: (是否跌倒, 边界框, 关键点, 检测信息)
        """
        results = self.model(frame, verbose=False)
        
        fall_detected = False
        bbox = None
        keypoints = None
        info = ""
        
        for result in results:
            boxes = result.boxes
            kpts = result.keypoints
            
            if boxes is None or kpts is None:
                continue
                
            for i, box in enumerate(boxes):
                if box.conf < 0.5:
                    continue
                    
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = x2 - x1
                h = y2 - y1
                
                keypoints_data = kpts.data[i].cpu().numpy()
                
                bbox = (int(x1), int(y1), int(x2), int(y2))
                keypoints = keypoints_data
                
                is_fall, reason = self._analyze_fall(w, h, keypoints_data)
                fall_detected = is_fall
                info = reason
                
                break
                
        return fall_detected, bbox, keypoints, info
        
    def _analyze_fall(self, w, h, keypoints):
        """
        基于几何规则分析跌倒
        """
        aspect_ratio = w / h if h > 0 else 0
        
        conf_threshold = 0.5
        
        shoulder_l = keypoints[5] if keypoints[5][2] > conf_threshold else None
        shoulder_r = keypoints[6] if keypoints[6][2] > conf_threshold else None
        hip_l = keypoints[11] if keypoints[11][2] > conf_threshold else None
        hip_r = keypoints[12] if keypoints[12][2] > conf_threshold else None
        
        shoulder_center = None
        hip_center = None
        
        if shoulder_l and shoulder_r:
            shoulder_center = ((shoulder_l[0] + shoulder_r[0]) / 2, 
                             (shoulder_l[1] + shoulder_r[1]) / 2)
        elif shoulder_l:
            shoulder_center = (shoulder_l[0], shoulder_l[1])
        elif shoulder_r:
            shoulder_center = (shoulder_r[0], shoulder_r[1])
            
        if hip_l and hip_r:
            hip_center = ((hip_l[0] + hip_r[0]) / 2, 
                         (hip_l[1] + hip_r[1]) / 2)
        elif hip_l:
            hip_center = (hip_l[0], hip_l[1])
        elif hip_r:
            hip_center = (hip_r[0], hip_r[1])
        
        is_fall = False
        reason = ""
        
        if aspect_ratio > 1.2:
            is_fall = True
            reason = f"宽高比异常: {aspect_ratio:.2f}"
        elif shoulder_center and hip_center:
            dx = shoulder_center[0] - hip_center[0]
            dy = shoulder_center[1] - hip_center[1]
            
            if dy == 0:
                angle = 90.0
            else:
                angle = abs(np.degrees(np.arctan2(abs(dx), abs(dy))))
                
            if angle > 60:
                is_fall = True
                reason = f"躯干倾斜角度: {angle:.1f}°"
            else:
                reason = f"躯干角度正常: {angle:.1f}°"
        else:
            reason = "关键点检测不完整"
            
        return is_fall, reason


class AlertSystem:
    """
    报警系统 - TTS 语音播报（女性声音）
    """
    def __init__(self):
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
            
            voices = self.engine.getProperty('voices')
            if voices:
                for voice in voices:
                    voice_name = voice.name.lower()
                    if 'female' in voice_name or 'huihui' in voice_name or 'xiaoxiao' in voice_name or 'chinese' in voice_name:
                        self.engine.setProperty('voice', voice.id)
                        print(f"已设置女性声音: {voice.name}")
                        break
                else:
                    self.engine.setProperty('voice', voices[0].id)
                    print(f"使用默认声音: {voices[0].name}")
            
            self.enabled = True
            print("TTS 引擎初始化完成")
        except Exception as e:
            print(f"TTS 初始化失败: {e}")
            self.enabled = False
            
    def speak(self, text, blocking=False):
        if not self.enabled:
            print(f"[TTS] {text}")
            return
            
        print(f"[TTS] {text}")
        self.engine.say(text)
        if blocking:
            self.engine.runAndWait()
        
    def speak_async(self, text):
        """
        异步语音播报，不阻塞主线程
        """
        if not self.enabled:
            print(f"[TTS] {text}")
            return
            
        thread = threading.Thread(target=self.speak, args=(text, True))
        thread.start()
        
    def alert_fall(self, suggestion=None):
        if suggestion:
            self.speak_async(suggestion)
        else:
            self.speak_async("检测到跌倒，正在为您分析情况，请不要随意移动。")


class InteractionMode:
    """
    交互模式 - 跌倒后进入的紧急对话模式
    """
    def __init__(self, alert_system, vlm_analyzer):
        self.alert_system = alert_system
        self.vlm_analyzer = vlm_analyzer
        self.active = False
        self.fall_frame = None
        self.interaction_count = 0
        
    def activate(self, frame):
        self.active = True
        self.fall_frame = frame.copy()
        self.interaction_count = 0
        print("\n" + "="*50)
        print("进入紧急交互模式")
        print("="*50 + "\n")
        
    def deactivate(self):
        self.active = False
        self.fall_frame = None
        print("\n" + "="*50)
        print("退出紧急交互模式，恢复正常监控")
        print("="*50 + "\n")
        
    def get_suggestion_from_vlm(self):
        """
        从 VLM 获取急救建议
        """
        print("正在向大模型请求急救建议...")
        response = self.vlm_analyzer.analyze_fall(self.fall_frame)
        
        try:
            data = json.loads(response)
            suggestion = data.get("suggestion", "检测到跌倒，请不要随意移动。")
            analysis = data.get("analysis", "")
            risk = data.get("risk", "")
            severity = data.get("severity", "medium")
            
            print(f"\n[VLM 分析结果]")
            print(f"严重程度: {severity}")
            print(f"姿态分析: {analysis}")
            print(f"环境风险: {risk}")
            print(f"建议: {suggestion}\n")
            
            return suggestion, analysis, risk, severity
            
        except json.JSONDecodeError:
            print(f"[VLM 原始响应]: {response}")
            return "检测到跌倒，请不要随意移动。", "", "", "medium"


def main():
    
    # ========== 视频源选择 ==========
    # 选项 1: 本地视频文件（用于 MVP 演示，当前激活）
    video_path = r"d:\reps\ai_for_detection\ai-fall-detection\1272297698.mp4"
    
    # 选项 2: 实时摄像头（用于真实场景，后续取消注释即可）
    # video_path = 0  # 0 表示默认摄像头
    # ==================================
    
    simulator = VideoStreamSimulator(video_path, loop=True)
    detector = FallDetector()
    alert_system = AlertSystem()
    
    vlm_analyzer = VLMAnalyzer(
        api_key="ms-f7c7041f-6807-4267-9236-dfc17a724ba0"
    )
    
    interaction_mode = InteractionMode(alert_system, vlm_analyzer)
    
    simulator.open()
    
    last_alert_time = 0
    cooldown_seconds = 30
    
    consecutive_fall_frames = 0
    fall_threshold_frames = 5
    
    print("\n" + "="*50)
    print("跌倒检测系统启动")
    print("按 'q' 退出")
    print("按 's' 暂停/继续（仅视频模式）")
    print("按 'e' 手动进入/退出交互模式")
    print("="*50 + "\n")
    
    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("用户请求退出")
                break
                
            if key == ord('s'):
                is_paused = simulator.toggle_pause()
                if not simulator.is_camera:
                    print(f"视频流: {'暂停' if is_paused else '继续'}")
                
            if key == ord('e'):
                if interaction_mode.active:
                    interaction_mode.deactivate()
                    alert_system.speak_async("恢复正常监控。")
                else:
                    frame, _ = simulator.read_frame()
                    if frame is not None:
                        interaction_mode.activate(frame)
                        suggestion, _, _, _ = interaction_mode.get_suggestion_from_vlm()
                        alert_system.speak_async(suggestion)
            
            if interaction_mode.active:
                interaction_mode.interaction_count += 1
                if interaction_mode.interaction_count > 100:
                    interaction_mode.deactivate()
                    alert_system.speak_async("恢复正常监控。")
                    
                frame, ret = simulator.read_frame()
                if not ret:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    
                display_frame = frame.copy()
                
                cv2.putText(display_frame, "EMERGENCY MODE ACTIVE", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(display_frame, "Press 'e' to exit", (20, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Fall Detection System', display_frame)
                continue
                
            frame, ret = simulator.read_frame()
            if not ret:
                time.sleep(0.01)
                continue
                
            fall_detected, bbox, keypoints, info = detector.detect(frame)
            
            display_frame = frame.copy()
            
            if bbox:
                x1, y1, x2, y2 = bbox
                
                if fall_detected:
                    consecutive_fall_frames += 1
                    color = (0, 0, 255)
                    label = "FALL DETECTED"
                    
                    if consecutive_fall_frames >= fall_threshold_frames:
                        current_time = time.time()
                        if current_time - last_alert_time > cooldown_seconds:
                            print(f"\n{'='*50}")
                            print(f"检测到跌倒！原因: {info}")
                            print(f"连续跌倒帧数: {consecutive_fall_frames}")
                            print(f"{'='*50}\n")
                            
                            interaction_mode.activate(frame)
                            suggestion, analysis, risk, severity = interaction_mode.get_suggestion_from_vlm()
                            alert_system.speak_async(suggestion)
                            
                            last_alert_time = current_time
                else:
                    consecutive_fall_frames = max(0, consecutive_fall_frames - 1)
                    color = (0, 255, 0)
                    label = "NORMAL"
                    
                    if consecutive_fall_frames > 0 and consecutive_fall_frames < fall_threshold_frames:
                        label = f"CHECKING ({consecutive_fall_frames}/{fall_threshold_frames})"
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(display_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                cv2.putText(display_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(display_frame, info, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            fps_text = f"FPS: {int(1 / (time.time() - getattr(main, 'last_frame_time', time.time()))):d}"
            main.last_frame_time = time.time()
            cv2.putText(display_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Fall Detection System', display_frame)
            
    except KeyboardInterrupt:
        print("\n程序被中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        simulator.release()
        cv2.destroyAllWindows()
        print("程序结束")


if __name__ == "__main__":
    main()
