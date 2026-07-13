from openai import OpenAI

# 初始化DeepSeek客户端
client = OpenAI(
    api_key="sk-bf1d51b6d2d848f8a0eed686d56edfee",  # 替换成你的Key
    base_url="https://api.deepseek.com/v1"
)

# 测试连接
try:
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "你好，DeepSeek！"}]
    )
    print("✅ 连接成功！返回内容：")
    print(response.choices[0].message.content)
except Exception as e:
    print("❌ 连接失败，错误信息：", e)