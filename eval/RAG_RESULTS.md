# RAG Retrieval Evaluation

## Latest run

- Embedding model: `embedding-3`
- Index: 25 document chunks
- Dataset: 35 manually labeled retrieval questions
- Top-k: 3
- Recall@3: 35/35 (100.0%)
- Top-1 accuracy: 30/35 (85.7%)
- MRR: 0.9190
- Scoring: a hit is counted when at least one human-labeled relevant chunk appears in the top-k results.

| Query | Expected chunk IDs | Retrieved chunk IDs | Hit rank |
|---|---|---|---:|
| 真实硬件如何采集温湿度？ | 0, 1, 18 | 0, 7, 1 | 1 |
| ESP32 怎样通过 WiFi 和 MQTT 上报数据？ | 0, 1, 18 | 1, 18, 0 | 1 |
| 没有硬件时能否使用模拟设备？ | 1, 5, 6 | 1, 0, 17 | 1 |
| 项目使用什么数据库，保存了哪些表？ | 1, 23 | 23, 19, 13 | 1 |
| Flask 和 Chart.js 大屏展示哪些内容？ | 0, 1 | 4, 1, 9 | 2 |
| 告警系统有哪些功能？ | 2, 23 | 2, 14, 19 | 1 |
| MQTT 断线之后怎么重连？ | 2, 23 | 15, 2, 5 | 2 |
| 断网期间数据如何缓存和恢复补发？ | 2, 23 | 23, 2, 19 | 1 |
| 支持哪些数据导出和日报功能？ | 2, 20, 21 | 2, 19, 14 | 1 |
| 项目从传感器到网页的数据链路是怎样的？ | 0, 3, 4, 5, 6, 22 | 22, 13, 14 | 1 |
| 云端 Broker 和 Flask 大屏如何连接？ | 3, 4, 5, 6, 9 | 16, 1, 3 | 3 |
| 项目需要哪些硬件元件？ | 7 | 7, 19, 0 | 1 |
| DHT11 如何接线？ | 8, 18 | 18, 8, 15 | 1 |
| 外接 LED 的接线是什么？ | 8 | 8, 18, 15 | 1 |
| 设备端使用了哪些技术？ | 8, 9 | 8, 11, 17 | 1 |
| 项目使用了哪几种 MQTT Broker？ | 9, 16 | 9, 5, 22 | 1 |
| 服务端数据存储使用什么技术？ | 9, 23 | 9, 19, 13 | 1 |
| 服务端使用什么语言和框架？ | 9 | 9, 16, 13 | 1 |
| 项目的推荐开发环境是什么？ | 10 | 10, 19, 7 | 1 |
| 设备端主程序和 MQTT 客户端库分别是什么文件？ | 11, 12 | 12, 11, 15 | 1 |
| 网页大屏由哪个文件实现？ | 12, 13 | 16, 18, 13 | 3 |
| 如何启动网页大屏？ | 16 | 16, 18, 0 | 1 |
| 如何启动本地 Broker、模拟设备和数据采集？ | 16, 17 | 17, 15, 14 | 1 |
| ESP32 上传程序和配置 WiFi 的步骤是什么？ | 17, 18 | 18, 11, 3 | 1 |
| DHT11 的 DATA 引脚接到哪个 GPIO？ | 18 | 18, 8, 7 | 1 |
| 如何使用命令行查看数据库统计或告警记录？ | 20 | 20, 14, 19 | 1 |
| 如何实时过滤告警和推送日志？ | 21 | 21, 19, 14 | 1 |
| 端到端链路设计包含哪些部分？ | 0, 22 | 22, 19, 23 | 1 |
| MQTT 使用了哪些主题、QoS 或遗嘱消息设计？ | 22 | 22, 9, 5 | 1 |
| DHT11 编程需要关注哪些时序和采样限制？ | 22 | 22, 18, 7 | 1 |
| GPIO 输入如何处理悬空抖动和按键去抖？ | 23 | 23, 19, 7 | 1 |
| 告警冷却机制怎么避免告警风暴？ | 2, 23 | 23, 2, 19 | 1 |
| 数据库如何支持表结构演进？ | 23 | 1, 23, 19 | 2 |
| WiFi 凭据和 Webhook 地址如何避免提交到仓库？ | 24 | 24, 18, 19 | 1 |
| 项目后续计划接入哪些功能？ | 24 | 24, 19, 0 | 1 |

## Current limitations

- The dataset is a small development regression set, not a production benchmark.
- The labels accept any relevant chunk, not a strict single gold answer.
- This run evaluates retrieval only, not answer faithfulness or citation correctness.
- The current index uses in-memory cosine plus IDF-weighted keyword retrieval for a small document set.
