from openai import OpenAI
import os

# 1. 配置客户端
# 这里的 api_key 填入你提供的: "ms-f7c7041f-6807-4267-9236-dfc17a724ba0"
client = OpenAI(
    api_key="ms-f7c7041f-6807-4267-9236-dfc17a724ba0", 
    base_url="https://api-inference.modelscope.cn/v1/"
)

# 2. 准备测试提示词
# 模拟 YOLO 已经检测到了跌倒，现在问 AI 该怎么办
prompt = "目前检测到一位老人在家中客厅跌倒，无法起身。请立即给出3条简短、核心的急救处理建议，供监护人参考。"

print("正在尝试连接模型...")

try:
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct", # 更换为 ModelScope 支持的 Qwen2.5 模型
        messages=[
            {
                'role': 'system',
                'content': '你是一个专业的医疗急救助手。回答要简练、精准，适合紧急情况阅读。'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ],
        stream=True # 保持流式输出
    )

    print("模型连接成功！回答如下：\n" + "-"*30)
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end='', flush=True)
    print("\n" + "-"*30)

except Exception as e:
    print(f"连接出错: {e}")