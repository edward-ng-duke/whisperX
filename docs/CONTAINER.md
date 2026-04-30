# WhisperX 容器使用文档

本仓库已经把 [whisperX](https://github.com/m-bain/whisperX) 包装成一个**完全离线的 HTTP 服务**，并提供 GPU / CPU 两套 Docker 镜像。本文档汇总：

- **实时说话人分离能否做？** → 见 [§7](#7-实时说话人分离重要)
- 两类镜像形态、三种启动方式、四个 HTTP 端点、环境变量、常见故障排查

接口字段语义请看 [API.md](API.md)；Python 侧的本地部署看 [../README.md](../README.md)。

---

## 1. TL;DR

| 能做的事 | 能不能 |
|---|---|
| 离线 ASR（faster-whisper） | ✅ |
| 词级时间戳（wav2vec2 / torchaudio 对齐） | ✅ |
| 说话人分离（pyannote `speaker-diarization-community-1`） | ✅ 但只能**整段离线**做 |
| 完全断网运行（`HF_HUB_OFFLINE=1`） | ✅ |
| **实时 / 流式说话人分离（边说边分）** | ❌ **不支持**，详见 §7 |

---

## 2. 镜像形态

| 形态 | Dockerfile / target | 镜像大小 | 适用场景 |
|---|---|---|---|
| **GPU bundled** | `Dockerfile.cuda --target=bundled` | ~55 GB | 生产首选，模型烤进镜像 |
| **CPU bundled** | `Dockerfile.cpu --target=bundled` | ~50 GB | 无 GPU 的部署 |
| GPU base | `Dockerfile.cuda --target=base` | ~7 GB | 开发，把宿主机 `./models` 挂进容器 |
| CPU base | `Dockerfile.cpu --target=base` | ~3 GB | 同上，CPU 版 |

`bundled` 把 `./models/` 整个 `COPY` 进镜像 → 启动后立即可用、零网请求。
`base` 只装代码和依赖 → 必须 `-v ./models:/app/models` 挂卷。

镜像默认运行 `uvicorn deploy.api.server:app --host 0.0.0.0 --port 8000`。

---

## 3. 启动方式（三选一）

### 3.1 docker compose（推荐）

前置一次性准备（详见 [../README.md](../README.md)）：

```bash
cp .env.example .env.local
chmod 600 .env.local
$EDITOR .env.local                            # 填 HF_TOKEN（仅首次下载用）
uv run python deploy/scripts/download_models.py  # 把模型下到 ./models/
```

构建并启动：

```bash
# GPU
docker compose --profile gpu build
docker compose --profile gpu up -d

# CPU
docker compose --profile cpu build
docker compose --profile cpu up -d
```

`docker-compose.yml` 已经声明：

- 端口 `8000:8000`
- GPU profile：`--gpus all` + `DEVICE=cuda`、`COMPUTE_TYPE=float16`、`WHISPER_MODEL=small`
- CPU profile：`DEVICE=cpu`、`COMPUTE_TYPE=int8`、`WHISPER_MODEL=tiny`
- `restart: unless-stopped`

可用环境变量覆盖：

```bash
WHISPER_MODEL=large-v3 LOG_LEVEL=DEBUG docker compose --profile gpu up -d
```

### 3.2 docker run（手工）

GPU bundled：

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e WHISPER_MODEL=small \
  whisperx-private:cuda-bundled
```

CPU bundled：

```bash
docker run --rm -p 8000:8000 \
  -e WHISPER_MODEL=tiny -e COMPUTE_TYPE=int8 \
  whisperx-private:cpu-bundled
```

### 3.3 base 镜像 + 宿主机挂模型

适合频繁更换模型 / 节约镜像层：

```bash
docker build -f Dockerfile.cuda --target=base -t whisperx-private:cuda-base .

docker run --rm --gpus all -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  whisperx-private:cuda-base
```

---

## 4. HTTP API 速查

来源：[../api/server.py](../api/server.py)。

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/health` | 健康检查、当前 device/compute_type、已加载模型 |
| `GET` | `/models` | 本地可用 whisper sizes、对齐语言、`diarization_ready` |
| `POST` | `/transcribe` | 完整音频 → segments + 词级时间戳 + （可选）speaker label |
| `POST` | `/diarize` | 完整音频 → 仅说话人时间段（不含转写文本） |
| `GET` | `/docs` | Swagger UI（FastAPI 自动） |

### 4.1 仅转写

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@./audio.mp3" \
  -F "model=large-v3" \
  -F "language=zh"
```

### 4.2 转写 + 说话人分离（已知 2 人）

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@./meeting.wav" \
  -F "model=large-v3" \
  -F "diarize=true" \
  -F "num_speakers=2"
```

### 4.3 仅做说话人分离（不要转写）

```bash
curl -X POST http://localhost:8000/diarize \
  -F "file=@./meeting.wav" \
  -F "min_speakers=2" -F "max_speakers=4"
```

完整字段语义、错误码、并发说明 → [API.md](API.md)。

---

## 5. 环境变量速查

| 变量 | 默认 | 说明 |
|---|---|---|
| `HF_TOKEN` | 空 | **仅首次下载模型**用（pyannote 是 gated 模型）；运行时不需要 |
| `DEVICE` | `cuda` | `cuda` 或 `cpu` |
| `COMPUTE_TYPE` | GPU `float16` / CPU `int8` | CTranslate2 计算精度，可选 `float16` / `int8_float16` / `int8` |
| `WHISPER_MODEL` | `small`（GPU）/ `tiny`（CPU） | 缺省 whisper size |
| `API_HOST` | `0.0.0.0` | uvicorn bind |
| `API_PORT` | `8000` | uvicorn port |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `HF_HUB_OFFLINE` | `1` | 镜像内强制离线（已自动设置） |
| `TRANSFORMERS_OFFLINE` | `1` | 同上（已自动设置） |
| `HF_HOME` | `/app/models/hf` | HF 缓存路径（已自动设置） |
| `TORCH_HOME` | `/app/models/torch` | torch.hub 缓存路径（已自动设置） |

---

## 6. 模型选择（GPU）

| 模型 | 显存（fp16） | 速度（RTX 4090） | 精度 | 适用 |
|---|---|---|---|---|
| `tiny` | <1 GB | ~70× 实时 | 低 | 快速预览 |
| `base` | ~1 GB | ~50× 实时 | 一般 | 短文本 |
| `small` | ~2 GB | ~30× 实时 | 中 | **默认推荐** |
| `medium` | ~5 GB | ~15× 实时 | 较高 | 长音频 |
| `large-v3` | ~10 GB | ~8× 实时 | 最高 | 重要内容 / 中英混杂 |

> CPU 模式建议 `tiny` / `base` / `small` + `COMPUTE_TYPE=int8`，更大模型在 CPU 上不实用。

---

## 7. 实时说话人分离（重要）

### 结论：whisperx **不支持** 实时 / 流式说话人分离

| 证据 | 位置 |
|---|---|
| `DiarizationPipeline.__call__` 只接受**完整音频**（路径或完整 numpy 数组），无 chunk/stream 接口 | [whisperx/diarize.py](../../whisperx/diarize.py) |
| 后端模型 `pyannote/speaker-diarization-community-1` 是离线 segmentation + **全局聚类**，本身就不是流式 | 同上 |
| HTTP API `/transcribe` 与 `/diarize` 都是 multipart 完整文件上传 | [../api/server.py](../api/server.py) |
| README 里 "70× realtime" 指**推理比实时快 70 倍**，不是流式 | 上游 README |

### 如果你需要边说边分，可考虑：

| 方案 | 定位 | 代价 |
|---|---|---|
| **NVIDIA NeMo Sortformer streaming** | 真正端到端流式分离（2024 发布） | 引入 NeMo 框架 + 不同模型权重，与 whisperx 无关 |
| **pyannote.audio `OnlineSpeakerDiarization`** | pyannote 自家的滚动缓冲方案 | 精度低于离线，本质仍是滑窗近似 |
| **diart**（基于 pyannote） | 第三方 pip 包，开箱即用 | 同上：滑窗 + 增量聚类的近似实时 |
| 自己用 `silero-vad` + 滑窗 + 增量聚类 | 不引入新框架 | 工程量大，效果差于专门的流式模型 |

### 准实时变通方案（延迟数秒可接受时）

每隔 N 秒（例 5 s）把累积音频整体送进现有 `/transcribe`。**注意**：

- 每次返回的 `SPEAKER_00`/`SPEAKER_01` 仅在该次切片内有效，**跨切片不会自动对齐到同一人**
- 要做跨切片对齐，需要拿 speaker embedding 做匹配。whisperx 的 `DiarizationPipeline` 已支持 `return_embeddings=True`（CLI 对应 `--speaker_embeddings`），但**当前 HTTP API 未暴露这个字段** — 见 [whisperx/diarize.py](../../whisperx/diarize.py)。如需启用，需要改 [../api/server.py](../api/server.py) 和 [../api/pipeline.py](../api/pipeline.py)。

---

## 8. 离线性 / 故障排查

### 离线验证

```bash
sudo unshare -n bash -c \
  'curl -F file=@/tmp/jfk.flac -F model=small http://localhost:8000/transcribe'
```

`unshare -n` 在网络命名空间隔离下执行，仍能完成转写即证明镜像离线 OK。

### 常见错误

| 现象 | 原因 / 解法 |
|---|---|
| `400 No default align-model for language: xx` | 该语言对齐模型未预下载，重跑 `download_models.py --languages xx` |
| `400 empty upload` | 上传文件大小为 0，检查客户端 |
| `500 model file missing: ...` | `./models/` 缺少所需模型，重跑 `download_models.py` |
| 启动时 401 拉模型失败 | `HF_TOKEN` 没填，或没在 HF 上 Accept pyannote 协议 |
| GPU OOM | 降低 `batch_size`（form 字段）或换更小的 `WHISPER_MODEL` |
| 多并发请求很慢 | 进程内有锁，单实例同时只跑 1 个请求；高并发用 `uvicorn --workers N`，注意显存 ×N |

容器启动时间：首次约 10–30 秒（加载模型到显存）；之后驻留。

---

## 9. 相关文件

- [Dockerfile.cuda](../../Dockerfile.cuda) / [Dockerfile.cpu](../../Dockerfile.cpu)
- [docker-compose.yml](../../docker-compose.yml)
- [.env.example](../../.env.example)
- [../api/server.py](../api/server.py) — FastAPI 入口
- [../api/pipeline.py](../api/pipeline.py) — whisperx 调用封装
- [../scripts/download_models.py](../scripts/download_models.py) — 模型预下载
- [API.md](API.md) — 端点详细字段语义
- [../README.md](../README.md) — 本地（非容器）部署
