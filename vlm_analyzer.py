import base64
import io
from openai import OpenAI


class VLMAnalyzer:
    """
    视觉大模型分析器
    使用 Qwen-VL (QVQ) 进行场景理解和急救建议生成
    """
    def __init__(self, api_key, base_url="https://api-inference.modelscope.cn/v1/"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = "Qwen/QVQ-72B-Preview"
        print("VLM 分析器初始化完成")
        
    def encode_image(self, image):
        """
        将 OpenCV 图像编码为 Base64
        """
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def analyze_fall(self, image):
        """
        分析跌倒场景并生成急救建议
        """
        image_b64 = self.encode_image(image)
        
        system_prompt = """你是一名专业的急救医生，正在通过监控观察一位刚跌倒的老人。
你的任务是分析图片并给出针对性的急救建议。

请按照以下格式输出：
1. 跌倒姿态分析：描述老人是侧身倒地、面部朝下还是仰面，头部是否着地
2. 环境风险评估：判断地面是否湿滑、周围有无尖锐物体
3. 急救建议：用口语化、简短的话语给出建议（用于语音播报）

建议应该根据严重程度区分：
- 严重（头部着地/撞击剧烈）：强调不要移动，保持平躺
- 中等：询问意识状态，检查是否有剧痛
- 轻微：提醒缓慢起身，注意平衡

输出格式（JSON）：
{
    "severity": "high/medium/low",
    "analysis": "跌倒姿态分析",
    "risk": "环境风险评估",
    "suggestion": "急救建议（口语化，适合语音播报）"
}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "分析这张图片中老人的跌倒情况，给出急救建议。"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                        ]
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"VLM 分析失败: {e}")
            return '{"severity": "medium", "analysis": "无法分析", "risk": "未知", "suggestion": "检测到跌倒，请不要随意移动，正在联系家人。"}'
    
    def analyze_with_feedback(self, image, user_feedback):
        """
        结合老人反馈进行二次分析
        """
        image_b64 = self.encode_image(image)
        
        system_prompt = """你是一名专业的急救医生。
老人已经跌倒，并给出了反馈。请结合反馈给出针对性的指导。

输出格式：
{
    "suggestion": "针对老人反馈的急救建议（口语化，适合语音播报）"
}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"老人说：{user_feedback}。请给出指导建议。"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                        ]
                    }
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"VLM 反馈分析失败: {e}")
            return '{"suggestion": "请保持冷静，我已通知家属和急救中心。"}'


import cv2
