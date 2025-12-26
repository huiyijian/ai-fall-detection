from openai import OpenAI
import os

# 配置 API Key
API_KEY = "ms-f7c7041f-6807-4267-9236-dfc17a724ba0"
BASE_URL = "https://api-inference.modelscope.cn/v1/"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def test_model(model_name, description):
    print(f"\n{'='*20}\n正在测试模型: {model_name} ({description})")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {'role': 'system', 'content': '你是一个急救助手。'},
                {'role': 'user', 'content': '检测到老人跌倒，请简短回答3个急救步骤。'}
            ],
            stream=False
        )
        print(f"✅ 连接成功！")
        print(f"回复内容: {response.choices[0].message.content[:50]}...")
        return True
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False

if __name__ == "__main__":
    print("开始模型可用性测试...")
    
    # 1. 测试用户指定的 Omni 模型
    test_model("Qwen/Qwen3-Omni-30B-A3B-Instruct", "用户指定的 Omni 模型")
    
    # 2. 测试替代视觉模型 (QVQ)
    test_model("Qwen/QVQ-72B-Preview", "替代视觉模型 (QVQ)")
    
    # 3. 测试替代文本模型 (Qwen2.5)
    test_model("Qwen/Qwen2.5-72B-Instruct", "替代文本模型 (Qwen2.5)")
