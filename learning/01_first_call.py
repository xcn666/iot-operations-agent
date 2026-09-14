# 01_first_call.py - 第一次调用大模型 API
# 使用前: cp config.example.py config.py 然后填入你的 API Key
import json
import requests

try:
    from config import API_KEY, API_URL, MODEL
except ImportError:
    print("找不到 config.py，请先执行:")
    print("  cd ~/iot-lab/ai")
    print("  cp config.example.py config.py")
    print("  nano config.py   # 填入你的 API Key")
    raise SystemExit(1)

if not API_KEY or "你的" in API_KEY:
    print("config.py 里的 API_KEY 还没填，请编辑 config.py 填入真实 Key")
    raise SystemExit(1)

headers = {
    "Authorization": "Bearer " + API_KEY,
    "Content-Type": "application/json",
}
payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "你是一个乐于助人的助手，回答要简洁。"},
        {"role": "user", "content": "用一句话介绍什么是物联网"},
    ],
    "temperature": 0.7,
}

print("正在调用大模型:", MODEL)
print("接口地址:", API_URL)
resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
print("HTTP 状态码:", resp.status_code)

if resp.status_code != 200:
    print("\n调用失败，服务器返回：")
    print(resp.text[:600])
    print("\n常见原因：API Key 错误 / 未实名 / 余额不足 / 模型名不对")
    raise SystemExit(1)

result = resp.json()
answer = result["choices"][0]["message"]["content"]
usage = result.get("usage", {})

print("\n===== 模型回答 =====")
print(answer)
print("\n===== 本次用量 =====")
print("输入 tokens:", usage.get("prompt_tokens"))
print("输出 tokens:", usage.get("completion_tokens"))
print("合计 tokens:", usage.get("total_tokens"))
