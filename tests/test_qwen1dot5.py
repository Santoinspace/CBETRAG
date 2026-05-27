from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    # model="/models/qwen",
    model='/root/autodl-tmp/Qwen2.5-7B-Instruct',
    messages=[
        {"role": "user", "content": "你好"}
    ]
)

print(response.choices[0].message.content)