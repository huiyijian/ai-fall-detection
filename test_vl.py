from openai import OpenAI
import os

# 这里的 api_key 填入你提供的
client = OpenAI(
    api_key="ms-f7c7041f-6807-4267-9236-dfc17a724ba0",
    base_url="https://api-inference.modelscope.cn/v1/"
)

# 测试 Qwen2-VL (视觉模型)
# 注意：如果没有图片，VL模型也可以像LLM一样处理纯文本，但最好提供图片测试其多模态能力
# 这里我们只测试连接性，使用纯文本
print("正在尝试连接 Qwen2-VL 模型...")

try:
    response = client.chat.completions.create(
        model="Qwen/QVQ-72B-Preview", 
        messages=[
            {
                'role': 'system',
                'content': '你是专业的助手。'
            },
            {
                'role': 'user',
                'content': '你好，能听到我说话吗？'
            }
        ],
        stream=False
    )
    print("Qwen2-VL 连接成功！")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"Qwen2-VL 连接出错: {e}")
