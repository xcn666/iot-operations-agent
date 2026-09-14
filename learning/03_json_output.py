# 03_json_output.py - 让大模型输出结构化 JSON（AI 应用开发的核心技能）
import json
import requests

try:
    from config import API_KEY, API_URL, MODEL
except ImportError:
    print("请先: cp config.example.py config.py 并填入 API Key")
    raise SystemExit(1)

headers = {"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"}

# ============ 模拟一段真实传感器数据（来自你的 IoT 项目）============
sensor_data = """设备 esp32-dht11 最近 10 分钟读数：
温度(摄氏度): 28, 29, 31, 35, 36, 34, 30, 29, 28, 27
湿度(%): 55, 57, 60, 72, 80, 78, 65, 60, 58, 56"""

# ============ 关键：设计 system prompt ============
system_prompt = """你是一个物联网数据分析助手。
示例：
输入：温度 25,26,27；湿度 50,52,54
输出：{"status": "正常", "max_temp": 27, "max_hum": 54, "reason": "均未超过阈值"}

请分析用户提供的传感器数据。

JSON 格式必须严格如下：
{
  "status": "正常" 或 "异常",
  "max_temp": 最高温度(数字),
  "max_hum": 最高湿度(数字),
  "reason": "一句话说明判断依据"
}

判断规则：最高温度 > 40 摄氏度 或 最高湿度 > 90% 视为异常。"""

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sensor_data},
    ],
    "temperature": 0.1,      # 结构化输出时, 温度调低更稳定
}

print("正在让模型分析传感器数据...")
resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
print("HTTP 状态码:", resp.status_code)
if resp.status_code != 200:
    print(resp.text[:400]); raise SystemExit(1)

content = resp.json()["choices"][0]["message"]["content"]
print("\n===== 模型原始输出 =====")
print(content)

# ============ 处理常见坑：模型可能把 JSON 包在 ```json ... ``` 里 ============
text = content.strip()
if text.startswith("```"):
    print("\n[提示] 模型输出了 markdown 代码块，正在自动剥离...")
    text = text.split("```")[1]
    if text.lstrip().startswith("json"):
        text = text.lstrip()[4:]
    text = text.strip()

# ============ 解析 JSON（程序真正能用的部分）============
try:
    data = json.loads(text)
    print("\n===== 解析成功！程序可以这样用 =====")
    print("状态      :", data.get("status"))
    print("最高温度  :", data.get("max_temp"), "°C")
    print("最高湿度  :", data.get("max_hum"), "%")
    print("判断依据  :", data.get("reason"))

    # 实际应用里就能直接做判断了:
    if data.get("status") == "异常":
        print("\n>>> 触发告警！可以推送到钉钉/企业微信，或写入 alerts 表")
except json.JSONDecodeError as e:
    print("\n[解析失败]", e)
    print("原始输出不是合法 JSON，可以尝试：降低 temperature / 强化 prompt 约束")
