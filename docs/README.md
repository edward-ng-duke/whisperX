# WhisperX 私有化部署 — 速览

把 [whisperX](https://github.com/m-bain/whisperX) 包装成一个**完全离线的 HTTP 服务**：自动语音识别（ASR） + 词级时间戳对齐 + 说话人分离（Speaker Diarization），打包成 GPU / CPU 两套 Docker 镜像，断网可跑。

> 这是 `docs/` 目录的入口页。详细内容在右边 4 篇专题文档里，本页只做**一页速览**。

---

## 1. 能做 / 不能做

| 能力 | 支持？ |
|---|---|
| 离线 ASR（faster-whisper） | ✅ |
| 词级时间戳（wav2vec2 / torchaudio 对齐，16 种语言预下载） | ✅ |
| 整段离线说话人分离（pyannote-community-1） | ✅ |
| 完全断网运行（`HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`） | ✅ |
| **实时 / 流式说话人分离** | ❌ — 详见 [CONTAINER.md §7](CONTAINER.md#7-实时说话人分离重要) |

---

## 2. 30 秒快速启动

```bash
# 一次性准备（联网机器上）
cp .env.example .env.local && chmod 600 .env.local && $EDITOR .env.local  # 填 HF_TOKEN
uv sync && uv pip install -r deploy/requirements-api.txt
uv run python deploy/scripts/download_models.py                            # 拉模型 ~50 GB

# 跑起来
docker compose --profile gpu up -d                                         # CPU: --profile cpu
curl http://localhost:8000/health
```

完整流程（含先决条件、镜像构建、离线分发） → [docker_deploy.md](docker_deploy.md)。

---

## 3. HTTP 端点

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/health` | 健康检查、当前 device、已加载模型 |
| `GET` | `/models` | 本地可用 whisper sizes、对齐语言、`diarization_ready` |
| `POST` | `/transcribe` | 完整音频 → segments + 词级时间戳 +（可选）speaker label |
| `POST` | `/diarize` | 完整音频 → 仅说话人时间段 |
| `GET` | `/docs` | Swagger UI（FastAPI 自动） |

请求字段、响应结构、错误码 → [API.md](API.md)。

---

## 4. 关键环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `DEVICE` | `cuda` | `cuda` 或 `cpu` |
| `COMPUTE_TYPE` | GPU `float16` / CPU `int8` | CTranslate2 计算精度 |
| `WHISPER_MODEL` | `small`（GPU）/ `tiny`（CPU） | 默认 Whisper size |
| `API_PORT` | `8000` | uvicorn 端口 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

完整列表（含 `HF_HOME` / `TORCH_HOME` / 离线开关等已自动设置项） → [API.md §7](API.md) / [docker_deploy.md §4.4.4](docker_deploy.md)。

---

## 5. 模型怎么选（GPU）

| 模型 | 显存（fp16） | 速度（RTX 4090） | 适用 |
|---|---|---|---|
| `tiny` | <1 GB | ~70× 实时 | 快速预览 / 弱算力 |
| `small` | ~2 GB | ~30× 实时 | **默认推荐** |
| `large-v3` | ~10 GB | ~8× 实时 | 重要内容 / 中英混杂 |

中间档 `base` / `medium` 与 CPU 模式建议见 [API.md §3](API.md)。

---

## 6. 说话人分离三层调参

| 层 | 参数 | 何时调 |
|---|---|---|
| HTTP 入参 | `num_speakers` / `min_speakers` / `max_speakers` | 已知人数 / 想约束聚类范围 |
| 模型超参 | `clustering.threshold` / `segmentation.min_duration_off` | 改聚类敏感度、合并短停顿 |
| 后处理 | `gap_tol` / `ASR_MIN_SEC`（仅 `qwen3_diarize.py`） | 调 turn 粒度 |

各层默认值、场景速查表、症状→旋钮映射 → [diarization.md](diarization.md)。

---

## 7. 常见踩坑 Top 5

| 现象 | 处理 |
|---|---|
| 拉 pyannote 时 401 / gated repo | `HF_TOKEN` 没填 / 没在 HF 上 Accept 协议 |
| `400 No default align-model for language: xx` | `uv run python deploy/scripts/download_models.py --languages xx` |
| GPU OOM | 表单字段 `batch_size` 调小，或换更小 `WHISPER_MODEL` |
| 首次 `/transcribe` 很慢 + 日志有 `Downloading...` | 离线缓存路径漏了 → [docker_deploy.md §7.3](docker_deploy.md) |
| 想边说边分（实时 diarize） | 不支持，可考虑 NeMo Sortformer / diart → [CONTAINER.md §7](CONTAINER.md#7-实时说话人分离重要) |

---

## 8. 文档地图

| 文档 | 看这篇当你想… |
|---|---|
| [API.md](API.md) | 写客户端 / 排错：HTTP 字段、响应结构、错误码、Python 库 / CLI |
| [CONTAINER.md](CONTAINER.md) | 选镜像形态、启动方式速查、实时分离结论与替代方案 |
| [diarization.md](diarization.md) | 说话人分离结果不理想、想调参 |
| [docker_deploy.md](docker_deploy.md) | 从零搭部署、构建镜像、内部实现（cuDNN / 对齐缓存 / NLTK）、离线分发、安全 |
| [../README.md](../README.md) | 看上游 whisperX 项目本身 |
| [../deploy/README.md](../deploy/README.md) | 本地（非容器）跑起来 |
