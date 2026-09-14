# 02_chat_cli.py - 命令行多轮对话助手（理解"对话历史"的作用）
import requests

try:
    from config import API_KEY, API_URL, MODEL
except ImportError:
    print("找不到 config.py，请先: cp config.example.py config.py 并填入 API Key")
    raise SystemExit(1)

if not API_KEY or "你的" in API_KEY:
    print("请先在 config.py 里填入真实的 API_KEY")
    raise SystemExit(1)

headers = {"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"}

# ⭐ 核心概念: messages 列表 = 对话历史
# 模型本身没有记忆, 每次都要把完整历史发过去
messages = [
    {"role": "system", "content": "你是一个专业的助手，回答简洁，用中文回答。"},
]

print("=" * 50)
print("AI 助手已启动（输入 exit 退出）")
print("试试问: 我刚才问了你什么？  ← 验证对话记忆")
print("=" * 50)

while True:
    try:
        user_input = input("\n你: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n再见！")
        break

    if user_input.lower() in ("exit", "quit", "q", "退出"):
        print("再见！")
        break
    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})

    try:
        resp = requests.post(
            API_URL, headers=headers,
            json={"model": MODEL, "messages": messages, "temperature": 0.7},
            timeout=60,
        )
        if resp.status_code != 200:
            print("[错误]", resp.status_code, resp.text[:200])
            messages.pop()
            continue

        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": answer})
        print("\nAI:", answer)
        print("    [对话轮数: {}]".format(len(messages) // 2))
    except Exception as e:
        print("[异常]", e)
        messages.pop()
