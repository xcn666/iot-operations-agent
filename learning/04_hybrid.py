# 04_hybrid.py - 正确架构：LLM 负责【提取】，代码负责【判断】
# 对比 03_json_output.py：那里让 LLM 又提取又判断 -> 数值比较会算错
import json
import requests

try:
    from config import API_KEY, API_URL, MODEL
except ImportError:
    print("请先: cp config.example.py config.py 并填入 API Key")
    raise SystemExit(1)

headers = {"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"}

sensor_data = """设备 esp32-dht11 最近 10 分钟读数：
温度(摄氏度): 28, 29, 31, 35, 36, 34, 30, 29, 28, 27
湿度(%): 55, 57, 60, 72, 80, 78, 65, 60, 58, 56"""

# ============ 关键改动：只让 LLM 做「提取」，不做判断 ============
system_prompt = """你是一个数据提取助手。

从用户提供的传感器数据中提取两个数值，**只输出 JSON**，不要输出解释文字，不要加 markdown 代码块。

输出格式：
{
  "max_temp": 最高温度(数字, 不带单位),
  "max_hum": 最高湿度(数字, 不带单位)
}

注意：你只需要提取数值，不要做任何判断。"""

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sensor_data},
    ],
    "temperature": 0.1,
}

print("步骤1: 让 LLM 提取数值...")
resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
if resp.status_code != 200:
    print("调用失败:", resp.status_code, resp.text[:300]); raise SystemExit(1)

content = resp.json()["choices"][0]["message"]["content"].strip()
print("LLM 原始输出:", content)

# 剥离可能的 markdown 代码块
if content.startswith("```"):
    content = content.split("```")[1]
    if content.lstrip().startswith("json"):
        content = content.lstrip()[4:]
    content = content.strip()

data = json.loads(content)
max_temp = data["max_temp"]
max_hum = data["max_hum"]
print("提取结果: 最高温度 =", max_temp, " 最高湿度 =", max_hum)

# ============ ⭐ 关键：判断交给代码（确定性逻辑，永远不会算错）============
THRESHOLD_TEMP = 30
THRESHOLD_HUM = 90

print("\n步骤2: 用代码做判断（可靠）")
if max_temp > THRESHOLD_TEMP or max_hum > THRESHOLD_HUM:
    status = "异常"
    reason = "最高温度 {}°C（阈值 {}）或最高湿度 {}%（阈值 {}）超限".format(
        max_temp, THRESHOLD_TEMP, max_hum, THRESHOLD_HUM)
else:
    status = "正常"
    reason = "最高温度 {}°C、最高湿度 {}%，均在阈值范围内".format(max_temp, max_hum)

print("判定结果:", status)
print("判断依据:", reason)

# ============ 步骤3：再让 LLM 做它擅长的事——生成人性化告警文案 ============
if status == "异常":
    print("\n步骤3: 让 LLM 生成告警文案（LLM 擅长的部分）")
    prompt2 = "请用一句专业、简洁的中文，为下面这条物联网告警写一条通知文案（不超过40字）：\n" + reason
    r2 = requests.post(API_URL, headers=headers,
                       json={"model": MODEL,
                             "messages": [{"role": "user", "content": prompt2}],
                             "temperature": 0.7}, timeout=60)
    if r2.status_code == 200:
        print("告警文案:", r2.json()["choices"][0]["message"]["content"].strip())
else:
    print("\n（数据正常，无需生成告警文案）")

print("\n===== 架构总结 =====")
print("LLM  : 提取数值 / 生成文案   (模糊任务)")
print("代码 : 比较阈值 / 判定异常   (精确任务)  ← 这才是正确分工")
