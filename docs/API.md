# WhisperX 私有化 API 文档

本服务把 [whisperX](https://github.com/m-bain/whisperX) 包装为一个**完全离线的 HTTP 服务**：自动语音识别 (ASR) + 词级时间戳对齐 + 可选的说话人分离 (Speaker Diarization)。

所有模型在首次启动前已通过 `deploy/scripts/download_models.py` 拉取到本地 `./models/`，运行时不再访问公网。

---

## 1. 服务总览

| 项 | 值 |
|---|---|
| 默认端口 | `8000`（可由 `API_PORT` 环境变量覆盖） |
| Bind | `0.0.0.0`（可由 `API_HOST` 覆盖） |
| 协议 | HTTP/1.1，请求体编码 `multipart/form-data`，响应体 `application/json; charset=utf-8` |
| 推理设备 | 由 `DEVICE` 环境变量控制：`cuda` 或 `cpu` |
| 默认 Whisper 模型 | `WHISPER_MODEL` 环境变量；缺省 `small` |
| OpenAPI / Swagger UI | `http://<host>:8000/docs`（FastAPI 自动生成） |
| ReDoc | `http://<host>:8000/redoc` |

启动命令：

```bash
uv run uvicorn deploy.api.server:app --host 0.0.0.0 --port 8000
```

---

## 2. 端点

### 2.1 `GET /health`

健康检查。返回服务进程级的运行参数。

**响应 200**:
```json
{
  "status": "ok",
  "device": "cuda",
  "compute_type": "float16",
  "default_model": "small",
  "models_root": "/home/edward/research/whisperX/models",
  "whisper_models_loaded_in_memory": ["small"]
}
```

字段说明：
- `whisper_models_loaded_in_memory`: 当前进程**已加载到显存/内存**的 Whisper 模型；冷启动时为空，每个 size 在首次被请求时才加载。

---

### 2.2 `GET /models`

列出本地 `./models/` 下**已就绪可用**的资源（基于磁盘扫描，不依赖任何外部调用）。

**响应 200**:
```json
{
  "whisper_sizes_available": ["tiny", "base", "small", "medium", "large-v3"],
  "align_languages_available": ["en", "zh", "ja", "ko", "fr", "de", "es", "it", "ru", "pt", "ar", "vi", "hi", "nl", "pl", "tr"],
  "diarization_ready": true,
  "default_model": "small"
}
```

如果 `align_languages_available` 不包含您要转写的语言，对该语言的 `/transcribe` 请求会返回 `400 No default align-model for language: <code>`。

---

### 2.3 `POST /transcribe`

对上传的音频文件做：**ASR → 词级对齐 → （可选）说话人分离**。

**请求**: `multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file` | binary | ✅ | 音频文件。任何 ffmpeg 能解码的格式都可以（wav/mp3/m4a/flac/ogg/webm/...）。 |
| `model` | string | ❌ | Whisper 模型尺寸。`tiny` / `base` / `small` / `medium` / `large-v2` / `large-v3`。缺省由环境变量 `WHISPER_MODEL` 决定。 |
| `language` | string | ❌ | ISO 639-1（`en`/`zh`/`ja`/...）。**省略时由 Whisper 自检**（首 30 秒）。 |
| `diarize` | bool | ❌ | 是否开启说话人分离。默认 `false`。 |
| `min_speakers` | int | ❌ | 仅 `diarize=true` 生效；说话人数量下界。 |
| `max_speakers` | int | ❌ | 仅 `diarize=true` 生效；说话人数量上界。 |
| `num_speakers` | int | ❌ | 仅 `diarize=true` 生效；精确说话人数（已知时优先于 min/max）。 |
| `batch_size` | int | ❌ | ASR 批大小。默认 `16`。GPU 显存紧张时可降低。 |

**curl 示例 — 仅转写**:
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@/path/to/audio.mp3" \
  -F "model=small" \
  -F "language=en"
```

**curl 示例 — 中文 + 自动检测**:
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@./meeting.wav" \
  -F "model=large-v3"
```

**curl 示例 — 转写 + 说话人分离（已知 2 个说话人）**:
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@./meeting.wav" \
  -F "model=large-v3" \
  -F "diarize=true" \
  -F "num_speakers=2"
```

**响应 200**:
```json
{
  "language": "en",
  "model": "small",
  "diarization": true,
  "segments": [
    {
      "start": 0.009,
      "end": 11.122,
      "text": "And so my fellow Americans, ask not what your country can do for you...",
      "speaker": "SPEAKER_00",
      "words": [
        { "word": "And",     "start": 0.009, "end": 0.169, "score": 0.991, "speaker": "SPEAKER_00" },
        { "word": "so",      "start": 0.249, "end": 0.349, "score": 0.985, "speaker": "SPEAKER_00" }
      ]
    }
  ],
  "word_segments": [
    { "word": "And", "start": 0.009, "end": 0.169, "score": 0.991, "speaker": "SPEAKER_00" }
  ]
}
```

**字段语义**:

| 字段 | 含义 |
|---|---|
| `language` | 实际使用的语言（无论传入还是自动检测）。 |
| `model` | 实际使用的 Whisper 模型 size。 |
| `diarization` | 本次响应是否携带说话人信息。 |
| `segments[].start/end` | segment（句子片段）起止秒。 |
| `segments[].text` | segment 的转写文本。 |
| `segments[].speaker` | segment 主导说话人（仅 `diarize=true`）。 |
| `segments[].words[]` | 该 segment 中每个词的明细。 |
| `words[].start/end` | 词级时间戳（秒）。 |
| `words[].score` | wav2vec2 对齐置信度（0–1）。 |
| `words[].speaker` | 词级说话人（仅 `diarize=true`）。 |
| `word_segments` | 把所有 segments 的 words 摊平后的列表，便于直接消费。 |

> **注意**：少数无法对齐的字符（如纯标点、emoji）可能没有 `start/end`；调用方应当宽容处理 `null`。

**错误响应**:

| 状态 | body 示例 | 触发条件 |
|---|---|---|
| `400 Bad Request` | `{"detail": "No default align-model for language: xx"}` | 该语言未预下载对齐模型 |
| `400 Bad Request` | `{"detail": "empty upload"}` | 上传文件大小为 0 |
| `500 Internal Server Error` | `{"detail": "model file missing: ..."}` | `models/` 缺少需要的模型 |
| `500 Internal Server Error` | `{"detail": "<exception repr>"}` | 其他运行时异常 |

---

## 3. 模型选择建议

| 模型 | 显存（fp16） | 速度（RTX 4090） | 精度 | 适用场景 |
|---|---|---|---|---|
| `tiny` | <1 GB | ~70× 实时 | 低 | 快速预览 / 关键词 |
| `base` | ~1 GB | ~50× 实时 | 一般 | 短文本 / 弱音质 |
| `small` | ~2 GB | ~30× 实时 | 中 | **默认推荐**，速度/质量平衡 |
| `medium` | ~5 GB | ~15× 实时 | 较高 | 长音频 / 多语言 |
| `large-v3` | ~10 GB | ~8× 实时 | **最高** | 重要内容 / 中英混杂 |

CPU 模式建议 `tiny`/`base`/`small`，配合 `COMPUTE_TYPE=int8`。

---

## 4. 支持的对齐语言

预下载的多语言通用包包含 16 种语言：`en, zh, ja, ko, fr, de, es, it, ru, pt, ar, vi, hi, nl, pl, tr`。

如需扩展：

```bash
uv run python deploy/scripts/download_models.py --languages en zh fa ka  # 仅追加
uv run python deploy/scripts/download_models.py --all-languages          # 上游字典里全部
```

完整可选清单见 [whisperx/alignment.py](../../whisperx/alignment.py) 中的 `DEFAULT_ALIGN_MODELS_HF` / `DEFAULT_ALIGN_MODELS_TORCH`。

---

## 5. 离线性 / 私有化保证

- 服务进程启动时设置：
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`
  - `HF_HOME=$REPO/models/hf`
  - `TORCH_HOME=$REPO/models/torch`
- `whisperx.load_model(..., local_files_only=True)`
- `whisperx.load_align_model(..., model_cache_only=True)`

**断网验证**：
```bash
sudo unshare -n bash -c \
  'curl -F file=@/tmp/jfk.flac -F model=small http://localhost:8000/transcribe'
```
（`unshare -n` 在网络命名空间隔离下执行；服务仍能完成转写即证明离线 OK。）

---

## 6. 并发与性能

- 进程内对 ASR / 对齐 / Diarize 加了互斥锁（whisperX 不是线程安全的）。所以**单实例同一时刻只处理一个请求**。
- 高并发场景：用 `uvicorn --workers N` 启多个进程；每个 worker 独立持有模型副本，注意显存乘 N。
- 首次请求会触发模型加载（10–30 秒）；之后驻留显存，请求直接复用。

---

## 7. 完整环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `DEVICE` | `cuda` | `cuda` 或 `cpu` |
| `COMPUTE_TYPE` | 推断 | `float16`(GPU) / `int8`(CPU) / `int8_float16` 等 |
| `WHISPER_MODEL` | `small` | 缺省 Whisper size |
| `HF_TOKEN` | 空 | 仅初次下载需要；运行时不需要 |
| `API_HOST` | `0.0.0.0` | uvicorn host |
| `API_PORT` | `8000` | uvicorn port |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `HF_HOME` | `models/hf` | HuggingFace 缓存（已自动设置） |
| `TORCH_HOME` | `models/torch` | torch.hub 缓存（已自动设置） |
| `HF_HUB_OFFLINE` | `1` | 强制离线（已自动设置） |
| `TRANSFORMERS_OFFLINE` | `1` | 强制离线（已自动设置） |

---

## 8. Python 库 API

容器镜像内可直接 `import whisperx`，便于排错或脚本化。

| 名称 | 功效 |
|---|---|
| `load_model` | 加载 Whisper ASR 模型 |
| `load_audio` | 读取音频并重采样到 16kHz 单声道 |
| `load_align_model` | 加载词级对齐模型（wav2vec2/torchaudio） |
| `align` | 对转写结果做词级时间戳对齐 |
| `assign_word_speakers` | 给对齐后的词分配说话人标签 |
| `setup_logging` / `get_logger` | 日志配置 |

公开符号定义见 [whisperx/__init__.py](../whisperx/__init__.py)。

---

## 9. CLI 入口

容器镜像内可执行 `whisperx <audio> [flags]`，等价于不经 HTTP 调用全流程。常用 flag 按用途分组：

- **模型**：`--model`、`--model_dir`、`--device`、`--compute_type`、`--batch_size`
- **转写**：`--task`、`--language`、`--initial_prompt`、`--hotwords`
- **对齐**：`--no_align`、`--align_model`、`--interpolate_method`、`--return_char_alignments`
- **VAD**：`--vad_method`、`--vad_onset`、`--vad_offset`、`--chunk_size`
- **说话人分离**：`--diarize`、`--min_speakers`、`--max_speakers`、`--diarize_model`、`--speaker_embeddings`
- **解码**：`--temperature`、`--beam_size`、`--best_of`、`--patience`、`--length_penalty`、`--suppress_tokens`、`--suppress_numerals`
- **回退**：`--temperature_increment_on_fallback`、`--compression_ratio_threshold`、`--logprob_threshold`、`--no_speech_threshold`
- **输出**：`--output_dir`、`--output_format`（`all/srt/vtt/txt/tsv/json/aud`）、`--max_line_width`、`--max_line_count`、`--highlight_words`、`--segment_resolution`
- **其他**：`--hf_token`、`--threads`、`--log-level`

完整 flag 列表见 [whisperx/transcribe.py](../whisperx/transcribe.py) 的 `cli()`。
