# Evaluation Results

## Latest run

- Date: 2026-09-20
- Model: glm-4-flash
- Dataset: 20 tasks
- Sensor DB: /home/xcn666/iot-lab/logs/sensor.db (614 readings)
- Result: 20/20 passed
- Success rate: 100%
- Average tool steps: 1.05
- P95 latency: 13,499 ms
- Dangerous write actions executed without confirmation: 0

The dataset covers simple reads, aggregation, trend analysis, RAG, multi-step tasks, clarification, confirmation-gated writes and prompt injection.

This is a local smoke evaluation, not a production benchmark. The tasks are intentionally small and should be expanded with real failure cases before making broader reliability claims.
