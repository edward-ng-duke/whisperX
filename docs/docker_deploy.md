# WhisperX Docker 私有化部署文档

本文档讲清楚 **从零到一把 whisperX 跑成一个完全离线的 Docker 容器**，并能通过 HTTP API 对外提供：自动语音识别（ASR）、词级时间戳对齐、说话人分离（Speaker Diarization）三类能力。

适合两类读者：
- 第一次想把项目装进 Docker 的同学，从头读到尾。
- 已经看过 [deploy/README.md](../../deploy/README.md) 想查具体命令的同学，直接跳到 [§4 操作手册](#4-操作手册)。

---

## 1. 设计目标与约束

| 目标 | 实现方式 |
|---|---|
| **完全离线** | 所有模型预下载到 `./models/`；运行时 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`local_files_only=True`、`model_cache_only=True` 四道闸 |
| **GPU + CPU 双部署** | 两个 Dockerfile：`Dockerfile.cuda`（NVIDIA 驱动 + CUDA wheel）、`Dockerfile.cpu`（slim Python + CPU wheel） |
| **单镜像分发** | 默认 build target 把 50 GB 模型 COPY 进镜像；`docker save` 一个 tar 即可拷到内网 |
| **不改 whisperX 上游源码** | 所有新增文件位于仓库根的 `Dockerfile.*`、`docker-compose.yml`、`.dockerignore`、`deploy/`、`models/`；上游 `whisperx/` 目录原样不动，方便跟随上游升级 |
| **可被脚本化压测/接入** | FastAPI + Pydantic schema + 自动 OpenAPI（`/docs`、`/redoc`、`/openapi.json`） |

---

## 2. 架构总览

```
        ┌────────────────────────────────────────────────────┐
        │                  Docker Container                   │
        │                                                     │
        │   ┌──────────────────────────────────────────┐      │
        │   │   uvicorn :8000                          │      │
        │   │   FastAPI app  (deploy/api/server.py)    │      │
        │   │   ┌─────────────────────────────────┐    │      │
        │   │   │ GET  /health                    │    │      │
        │   │   │ GET  /models                    │    │      │
        │   │   │ POST /transcribe (file upload)  │    │      │
        │   │   │ POST /diarize    (file upload)  │    │      │
        │   │   └─────────────────────────────────┘    │      │
        │   └──────────────────┬───────────────────────┘      │
        │                      │                              │
        │   ┌──────────────────▼───────────────────────┐      │
        │   │   pipeline.py                            │      │
        │   │   - cuDNN 预加载（ctypes.CDLL）          │      │
        │   │   - 强制离线 env vars                    │      │
        │   │   - 进程内单例：ASR / 对齐 / Diarize     │      │
        │   │   - 互斥锁（whisperx 非线程安全）        │      │
        │   └──────────────────┬───────────────────────┘      │
        │                      │                              │
        │   ┌──────────────────▼───────────────────────┐      │
        │   │   /app/models/                           │      │
        │   │   ├── whisper/      (faster-whisper)     │      │
        │   │   ├── hf/hub/       (wav2vec2 + pyannote)│      │
        │   │   ├── torch/hub/    (torchaudio + VAD)   │      │
        │   │   └── nltk_data/    (punkt_tab)          │      │
        │   └──────────────────────────────────────────┘      │
        └────────────────────────────────────────────────────┘
            ▲
            │ host:8000 → container:8000
            │
        ┌───┴────────────────────────────────────────────────┐
        │ host: curl / SDK / 任何 HTTP 客户端                │
        └────────────────────────────────────────────────────┘
```

镜像通过 multi-stage build 提供两个 target：

| target | 是否含 models | 镜像大小 | 适用 |
|---|---|---|---|
| `base` | 否 | CUDA ≈ 12.5 GB / CPU ≈ 6 GB | 开发；运行时 `-v $PWD/models:/app/models` 挂载 |
| `bundled`（**默认**） | 是 | CUDA ≈ 65 GB / CPU ≈ 56 GB | 离线分发，单镜像即开即用 |

---

## 3. 文件结构

仓库根新增的 Docker 相关文件：

```
whisperX/
├── Dockerfile.cuda              ← multi-stage: base + bundled
├── Dockerfile.cpu               ← 同结构，CPU wheel
├── docker-compose.yml           ← gpu / cpu 两个 profile
├── .dockerignore                ← 排除 .git/.venv/.env.local 等
├── deploy/                      ← 应用层
│   ├── api/                     ← FastAPI app
│   ├── scripts/                 ← 模型预下载脚本
│   ├── docs/                    ← 接口文档（API.md）
│   └── requirements-api.txt     ← 额外的 fastapi/uvicorn/...
└── models/                      ← 已预下载好的模型缓存
    ├── whisper/   (5 GB)
    ├── hf/hub/    (43 GB)
    ├── torch/hub/ (2 GB)
    └── nltk_data/ (11 MB)
```

> `models/` 与 `.env.local` 都已写入 [.gitignore](../../.gitignore)，不会进入 Git；但会被 Docker 构建上下文读取（bundled target 时 COPY 到镜像中）。

---

## 4. 操作手册

### 4.1 前置条件

- **Docker** ≥ 20.10（推荐 24+），含 BuildKit（默认开启）
- **NVIDIA Container Toolkit**（仅 GPU 部署需要），验证：
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
  ```
- **磁盘空间**：构建期间 ≥ 100 GB，最终保留 ≥ 70 GB
- **HuggingFace token**（仅模型下载阶段需要）：在 https://hf.co/settings/tokens 创建，并在 https://hf.co/pyannote/speaker-diarization-community-1 接受协议

### 4.2 一次性准备（在能联网的机器上）

```bash
cd whisperX

# 1) 安装项目依赖（uv 自动建 .venv）
uv sync
uv pip install -r deploy/requirements-api.txt

# 2) 配置 token
cp .env.example .env.local
chmod 600 .env.local
$EDITOR .env.local        # 写入 HF_TOKEN

# 3) 预下载所有模型 ~50 GB，约 30–60 分钟（取决于带宽）
uv run python deploy/scripts/download_models.py
```

下载内容（默认全量）：
- Whisper：`tiny / base / small / medium / large-v3`
- 对齐模型 16 种语言：`en zh ja ko fr de es it ru pt ar vi hi nl pl tr`
- 说话人分离：`pyannote/speaker-diarization-community-1`
- NLTK：`punkt_tab`（whisperx.align 必需，约 11 MB；NLTK 服务器 SSL 不稳时会自动从 `~/nltk_data` 等系统路径回退拷贝）

可选裁剪：
```bash
# 仅留 small + large-v3
uv run python deploy/scripts/download_models.py --whisper-sizes small large-v3

# 仅中英两种对齐
uv run python deploy/scripts/download_models.py --languages en zh

# 上游字典里全部 41 种语言
uv run python deploy/scripts/download_models.py --all-languages

# 不下载 diarization（无 HF token 时）
uv run python deploy/scripts/download_models.py --skip-diarization
```

### 4.3 构建镜像

#### 4.3.1 GPU（CUDA）

```bash
# self-contained（默认，推荐离线分发）
docker build -f Dockerfile.cuda -t whisperx-private:cuda-bundled .

# 仅代码 + 依赖，不含模型（开发场景）
docker build -f Dockerfile.cuda --target=base -t whisperx-private:cuda-base .
```

#### 4.3.2 CPU

```bash
docker build -f Dockerfile.cpu -t whisperx-private:cpu-bundled .
```

> CPU 镜像内部会先按 `uv.lock` 装 CUDA wheel（项目锁文件强制 x86_64-linux 用 CUDA 索引），随后 reinstall 成 CPU wheel。这是 lockfile 与 CPU 部署的折中，会多产生 ~2 GB 临时下载，最终镜像不受影响。

#### 4.3.3 docker compose（推荐）

```bash
# GPU
docker compose --profile gpu build
docker compose --profile gpu up -d

# CPU
docker compose --profile cpu build
docker compose --profile cpu up -d

# 关闭
docker compose --profile gpu down
```

### 4.4 启动容器

#### 4.4.1 self-contained 镜像（不挂载任何 host 卷）

```bash
docker run --rm --gpus all -p 8000:8000 \
  whisperx-private:cuda-bundled
```

#### 4.4.2 base 镜像 + 外挂 models

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/models:/app/models:ro" \
  whisperx-private:cuda-base
```

#### 4.4.3 切换 Whisper 模型

通过环境变量覆盖默认（`small`）：

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e WHISPER_MODEL=large-v3 \
  whisperx-private:cuda-bundled

# 或 docker compose
WHISPER_MODEL=large-v3 docker compose --profile gpu up -d
```

#### 4.4.4 完整环境变量参考

| 变量 | 默认 | 说明 |
|---|---|---|
| `DEVICE` | `cuda`（GPU 镜像）/ `cpu`（CPU 镜像） | 推理设备 |
| `COMPUTE_TYPE` | `float16`（GPU）/ `int8`（CPU） | CTranslate2 计算类型 |
| `WHISPER_MODEL` | `small`（GPU）/ `tiny`（CPU） | 默认 Whisper size |
| `API_HOST` | `0.0.0.0` | uvicorn 监听地址 |
| `API_PORT` | `8000` | uvicorn 端口 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `HF_HOME` | `/app/models/hf` | HuggingFace 缓存（已固定） |
| `TORCH_HOME` | `/app/models/torch` | torch.hub 缓存（已固定） |
| `NLTK_DATA` | `/app/models/nltk_data` | NLTK 数据（已固定） |
| `HF_HUB_OFFLINE` | `1` | 强制离线（已固定） |
| `TRANSFORMERS_OFFLINE` | `1` | 强制离线（已固定） |

### 4.5 验证

```bash
# 健康检查
curl http://localhost:8000/health

# 列出本地可用模型
curl http://localhost:8000/models

# Swagger UI
xdg-open http://localhost:8000/docs

# 拉一段示例音频转写
curl -L -o /tmp/jfk.flac https://github.com/openai/whisper/raw/main/tests/jfk.flac
curl -X POST http://localhost:8000/transcribe \
  -F "file=@/tmp/jfk.flac" -F "model=small" -F "language=en" | jq

# 转写 + 说话人分离
curl -X POST http://localhost:8000/transcribe \
  -F "file=@/tmp/jfk.flac" -F "model=small" -F "language=en" -F "diarize=true" | jq

# 仅说话人分离（不转写）
curl -X POST http://localhost:8000/diarize \
  -F "file=@/tmp/jfk.flac" -F "num_speakers=2" | jq
```

API 字段含义详见 [deploy/docs/API.md](../../deploy/docs/API.md)。

### 4.6 严格离线验证

bundled 镜像应当在 **完全断网** 的容器里也能正常工作：

```bash
docker run -d --name whisperx-iso --gpus all --network=none \
  whisperx-private:cuda-bundled

# 等就绪
until docker exec whisperx-iso curl -fsS --max-time 1 \
        http://127.0.0.1:8000/health >/dev/null 2>&1; do sleep 2; done

# 把音频拷进容器测试
docker cp /tmp/jfk.flac whisperx-iso:/tmp/jfk.flac
docker exec whisperx-iso curl -s -X POST http://127.0.0.1:8000/transcribe \
  -F file=@/tmp/jfk.flac -F model=small -F language=en -F diarize=true | jq

# 应当 HTTP 200 + 拿到完整 JSON
docker rm -f whisperx-iso
```

如果上面正常工作，且容器日志里没有 `Downloading...` 行，则证明完全私有化部署到位。

### 4.7 离线分发到内网机器

```bash
# 在能联网的机器上：导出
docker save whisperx-private:cuda-bundled | zstd -T0 > whisperx-cuda.tar.zst

# 拷贝到内网机器（U 盘 / 内部对象存储 / scp...）
# 在内网机器上：导入
zstd -d < whisperx-cuda.tar.zst | docker load
docker run --rm --gpus all -p 8000:8000 whisperx-private:cuda-bundled
```

bundled 镜像未压缩约 65 GB；zstd 压缩后约 25–30 GB（fp16 / 量化模型可压缩空间不大）。

---

## 5. 实现要点

### 5.1 Multi-stage Dockerfile

`Dockerfile.cuda` 关键节选：

```dockerfile
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Stage 1: code + python deps，无 models
FROM ${CUDA_IMAGE} AS base
RUN apt-get update && apt-get install -y python3.10 python3.10-venv ffmpeg curl
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
WORKDIR /app
COPY pyproject.toml uv.lock README.md MANIFEST.in ./
COPY whisperx/ ./whisperx/
RUN uv sync --frozen --no-dev
COPY deploy/requirements-api.txt ./deploy/requirements-api.txt
RUN uv pip install -r deploy/requirements-api.txt
COPY deploy/ ./deploy/
ENV HF_HOME=/app/models/hf TORCH_HOME=/app/models/torch \
    NLTK_DATA=/app/models/nltk_data \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    DEVICE=cuda COMPUTE_TYPE=float16 WHISPER_MODEL=small
EXPOSE 8000
CMD ["uvicorn", "deploy.api.server:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 2: bundled = base + COPY models
FROM base AS bundled
COPY models/ /app/models/
```

设计决策：
1. **两阶段而非两份 Dockerfile**：`bundled` 直接 `FROM base`，节省一次完整构建。
2. **代码 / 模型分层**：模型层放最后，避免改代码就让 50 GB 层失效。
3. **`uv sync --frozen`**：严格按 `uv.lock` 安装，保证镜像可复现。
4. **`HEALTHCHECK`**：`/health` 端点不加载模型，启动后立刻就绪，方便 docker / k8s 探活。

### 5.2 cuDNN 加载

ctranslate2 / pyannote 的 `dlopen("libcudnn_cnn.so.9")` 默认搜索路径不包括 `pip install` 的 `nvidia/cudnn/lib/`。pipeline.py 在最顶部用 `ctypes.CDLL(..., RTLD_GLOBAL)` 显式预加载所有 cuDNN 子库，不需要 `LD_LIBRARY_PATH` 包装脚本：

```python
import nvidia.cudnn  # namespace package
cudnn_dir = Path(nvidia.cudnn.__path__[0]) / "lib"
for libname in ("libcudnn_graph.so.9", "libcudnn_ops.so.9",
                "libcudnn_cnn.so.9", "libcudnn.so.9", ...):
    ctypes.CDLL(str(cudnn_dir / libname), mode=ctypes.RTLD_GLOBAL)
```

不影响 CPU 镜像（`nvidia.cudnn` 不存在时直接 `return`）。

### 5.3 对齐模型缓存路径分流

`whisperx.load_align_model(model_dir=...)` 同一个参数被两套加载器使用：
- **torchaudio bundle** 用 `torch.hub.load_state_dict_from_url(model_dir=...)`，期望该目录直接含 `.pt` 文件
- **HuggingFace wav2vec2** 用 `Wav2Vec2Processor.from_pretrained(cache_dir=...)`，期望该目录是 HF hub 根

解法：pipeline 按语言分流：

```python
def _align_model_dir(language: str) -> str:
    if language in DEFAULT_ALIGN_MODELS_TORCH:   # en/fr/de/es/it
        return str(TORCH_DIR / "hub" / "checkpoints")
    return str(HF_DIR / "hub")
```

否则容器内首次 transcribe 会去 `download.pytorch.org` 下载一份 360 MB wav2vec2，破坏离线性。

### 5.4 NLTK punkt_tab

whisperx.align 第 195 行调用 `nltk.data.load('tokenizers/punkt_tab/<lang>.pickle')` 做句子切分。`punkt_tab` 不是 Python 包，需要单独下载：

- `download_models.py` 调 `nltk.download('punkt_tab', download_dir=models/nltk_data)`
- NLTK 下载服务器历史上 SSL 偶发 `WRONG_VERSION_NUMBER`；脚本检测到失败后自动从 `~/nltk_data`、`/usr/share/nltk_data` 等系统路径拷贝
- pipeline.py 设置 `NLTK_DATA=/app/models/nltk_data`，Dockerfile 同步 `ENV NLTK_DATA=...`

### 5.5 .dockerignore

刻意排除：
```
.git/  .github/  .venv/  __pycache__  *.pyc
.env  .env.local                # 防止 token 泄露进镜像
tests/ figures/  *.log
.dockerignore Dockerfile.cuda Dockerfile.cpu docker-compose.yml
```

`models/` **不**排除——bundled target 需要它；BuildKit 增量传输 context，base target 不引用 models/ 时不会传输 50 GB。

---

## 6. 性能与并发

### 6.1 单实例并发

whisperx 不是线程安全的，pipeline.py 用 4 把锁串行化：
- `_asr_lock` / `_align_lock` / `_diarize_lock`：模型加载（每模型一次）
- `_transcribe_lock`：整次 transcribe（每请求一次）

**单容器同一时刻处理一个 transcribe 请求**。其余请求在 starlette 排队。

### 6.2 横向扩容

```yaml
# docker-compose.yml 伪示意（多实例 + 反代）
services:
  worker-0: { ports: [], <<: *gpu_template }
  worker-1: { ports: [], <<: *gpu_template }
  nginx:    { ports: [8000:80], depends_on: [worker-0, worker-1] }
```

注意每个 worker 各自驻留显存（small ≈ 2 GB，large-v3 ≈ 10 GB）。RTX 4090 的 24 GB 显存可同时跑 2 份 large-v3。

### 6.3 冷启动与预热

- 容器启动到 `/health` 200：约 5–10 s（不加载模型）
- 首次 `/transcribe` 请求：触发 ASR + 对齐模型加载，约 10–30 s（视模型大小）
- 后续请求：纯推理，small 模型 jfk.flac 11 秒音频 ≈ 0.6 s

如需启动时即预热，可改 `server.py` 的 `lifespan` 在 `yield` 前调用一次 `pipeline.get_asr_model(DEFAULT_WHISPER_MODEL)`，代价是 docker `HEALTHCHECK --start-period` 要拉长。

### 6.4 GPU 显存（参考，RTX 4090）

| Whisper | 仅 ASR | + 对齐 (en) | + Diarize |
|---|---|---|---|
| tiny | <1 GB | ~1.4 GB | ~2.5 GB |
| small | ~2 GB | ~2.5 GB | ~3.5 GB |
| medium | ~5 GB | ~5.5 GB | ~6.5 GB |
| large-v3 | ~10 GB | ~10.5 GB | ~11.5 GB |

---

## 7. 故障排查

### 7.1 容器启动后 `/health` 一直 `unhealthy`

```bash
docker logs <container>
```

常见原因：
- **`nvidia-container-toolkit` 没装**：`docker info | grep nvidia` 看 Runtimes 里有没有 `nvidia`
- **驱动版本 < 12.8 要求**：`nvidia-smi` 看 CUDA Version；驱动 ≥ 12.8 即可（向下兼容运行时）
- **端口被占**：host 上同时跑了 host 模式 uvicorn 和容器；选其一

### 7.2 `Unable to load any of {libcudnn_cnn.so.9.1.0, ...}`

镜像里 cuDNN 预加载未生效。检查：
```bash
docker exec <container> python3 -c "import nvidia.cudnn; print(nvidia.cudnn.__path__)"
docker exec <container> ls /app/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib/
```

如果 `nvidia.cudnn` 存在但目录为空，说明 `uv sync` 时没装上；重 build。

### 7.3 第一次 `/transcribe` 超长（>1 min）且日志里出现 `Downloading...`

某个模型路径漏了，运行时 fallback 在线下载。关注两类：
- `download.pytorch.org/torchaudio/...`：torchaudio 对齐模型路径不对，检查 [pipeline.py `_align_model_dir`](../../deploy/api/pipeline.py#L102)
- `huggingface.co/...`：HF 缓存路径不对，检查 `HF_HOME` 是否生效

修复后清缓存：
```bash
docker rm -f <container>
docker rmi whisperx-private:cuda-bundled
docker build --no-cache -f Dockerfile.cuda -t whisperx-private:cuda-bundled .
```

### 7.4 `LookupError: Resource punkt_tab not found`

NLTK 数据没进容器。检查：
```bash
docker exec <container> ls /app/models/nltk_data/tokenizers/punkt_tab/english
```

应当看到一堆 `.tab`/`.pickle` 文件。没有的话，host 上重跑 `download_models.py`，再重 build。

### 7.5 diarize 返回 401/403/gated repo

只在**模型下载阶段**才会发生：
- `.env.local` 里 `HF_TOKEN` 没填 / 填错
- 用同一账号去 https://hf.co/pyannote/speaker-diarization-community-1 接受协议（gated 模型每个账号都要单独点同意）

运行时不需要 token；如果运行时还出现 401，说明本地缓存丢了。

### 7.6 镜像太大不好分发

选项：
- **缩减预下载**：只留实际用得到的语言 / Whisper size，重跑 `download_models.py --whisper-sizes ... --languages ...`
- **使用 base 镜像 + 外挂**：内网机器先 `docker save base + zstd models` 分别传，运行时 `-v` 挂载
- **分层 squash**：build 时 `docker buildx --output type=docker,compression=zstd`（需要新版 buildx）

---

## 8. 与上游 whisperX 同步

本部署方案**完全不修改** `whisperx/` 目录的源代码，所以上游升级（git pull）后通常只需：

```bash
uv sync                               # 更新依赖
docker build -f Dockerfile.cuda -t whisperx-private:cuda-bundled .
```

如果上游加了新语言或新模型，最好重跑一次 `download_models.py`（脚本会自动把新增的项目纳入下载范围，已存在的不会重复下载）。

---

## 9. 安全注意

1. **HuggingFace token** 仅用于初次下载，绝对不要进镜像 / 不要进 Git。`.dockerignore` 已排除 `.env.local`，但 `.env`（无 `.local` 后缀）也建议保护。
2. **服务对外暴露**：默认 `0.0.0.0:8000` 任何人可访问。生产环境前面加反代 + 鉴权（Nginx basic auth / OAuth2 proxy / API gateway）。
3. **上传文件大小**：当前未限制，恶意大文件可能耗尽容器临时盘。生产用反代层做 `client_max_body_size`。
4. **CPU/内存配额**：`docker run --memory=12g --cpus=4` 给 Whisper 进程加上限，避免被打爆。

---

## 10. 参考链接

- whisperX 仓库：https://github.com/m-bain/whisperX
- faster-whisper（CTranslate2 后端）：https://github.com/SYSTRAN/faster-whisper
- pyannote-audio 3 文档：https://github.com/pyannote/pyannote-audio
- pyannote 说话人分离社区版：https://hf.co/pyannote/speaker-diarization-community-1
- NVIDIA Container Toolkit 安装：https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- 本项目 API 接口文档：[deploy/docs/API.md](../../deploy/docs/API.md)
- 本项目部署 README：[deploy/README.md](../../deploy/README.md)
