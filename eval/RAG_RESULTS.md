# RAG Retrieval Evaluation

## Latest run

- Date: 2026-09-26
- Embedding model: `embedding-3`
- Index: 32 document chunks
- Dataset: 10 manually labeled retrieval questions
- Top-k: 3
- Recall@3: 8/10 (80%)
- MRR: 0.6333
- Scoring: a hit is counted when at least one human-labeled relevant chunk appears in the top 3 results.

| Query | Expected chunk IDs | Retrieved chunk IDs | Hit |
|---|---|---|---:|
| MQTT 断线之后怎么重连？ | 2, 29 | 19, 7, 28 | No |
| 告警冷却机制怎么避免告警风暴？ | 2, 29 | 29, 2, 24 | Yes |
| 项目使用什么数据库，保存了哪些表？ | 1, 2, 12 | 24, 29, 14 | No |
| 真实硬件如何采集温湿度？ | 1, 10, 22 | 1, 0, 10 | Yes |
| ESP32 怎样通过 WiFi 和 MQTT 上报数据？ | 1, 5, 20 | 1, 5, 15 | Yes |
| 告警如何推送到钉钉或企业微信？ | 2, 18, 25 | 2, 18, 26 | Yes |
| 断网期间数据如何缓存和恢复补发？ | 2, 29 | 3, 24, 29 | Yes |
| MQTT 使用了哪些主题、QoS 或遗嘱消息设计？ | 28 | 28, 7, 12 | Yes |
| DHT11 的接线和 GPIO 是什么？ | 10, 11, 22 | 23, 22, 10 | Yes |
| Flask 和 Chart.js 大屏展示哪些内容？ | 1, 2, 12, 20 | 6, 2, 13 | Yes |

## Current limitations

- The dataset contains only 10 manually labeled questions and is intended as a development smoke evaluation.
- The labels accept any relevant chunk, not a strict single gold answer.
- This run evaluates retrieval only, not answer faithfulness or citation correctness.
- The current index still uses in-memory cosine retrieval and is suitable for a small document set.
