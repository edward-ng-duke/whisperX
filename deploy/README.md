# WhisperX 私有化部署 (Phase A: 本地)

把 [whisperX](https://github.com/m-bain/whisperX) 跑成一个**完全离线、零外网依赖**的 HTTP 服务，模型全部预下载到 `./models/`，方便后续直接 `COPY` 进 Docker 镜像。

## 目录布局

```
deploy/
├── api/
│   ├── server.py        FastAPI 应用
│   ├── pipeline.py      whisperx 调用封装 + 进程级模型缓存
│   └── schemas.py       Pydantic 响应模型
├── scripts/
│   └── download_models.py  一次性预下载所有模型
└── docs/
    └── API.md           API 接口文档
models/                  预下载的模型缓存（gitignored）
├── whisper/             faster-whisper 模型
├── hf/hub/              HuggingFace 缓存（对齐 + 说话人分离）
└── torch/hub/           torch.hub 缓存（torchaudio 对齐 + 可选 Silero VAD）
.env.example             环境变量模板
.env.local               真实 token（gitignored，已用 chmod 600 保护）
```

## 一次性准备

### 1. 安装系统依赖

```bash
sudo apt update && sudo apt install -y ffmpeg
nvidia-smi          # 如有 GPU；CPU 部署可跳过
```

### 2. 安装 Python 依赖

```bash
uv sync                                       # 装上游 whisperx + 所有 ML 依赖
uv pip install -r deploy/requirements-api.txt # 装 fastapi/uvicorn/python-multipart
```

### 3. 配置 .env.local

```bash
cp .env.example .env.local
chmod 600 .env.local
$EDITOR .env.local         # 填入你的 HF_TOKEN
```

> ⚠️ HF token 必须先在 https://hf.co/pyannote/speaker-diarization-community-1
> 用同一账号点 "Accept" 接受协议，否则 diarization 模型下载会 401。

### 4. 预下载所有模型

```bash
uv run python deploy/scripts/download_models.py
```

默认下载内容：
- Whisper：`tiny / base / small / medium / large-v3`
- 对齐模型 16 种语言：`en zh ja ko fr de es it ru pt ar vi hi nl pl tr`
- 说话人分离：`pyannote/speaker-diarization-community-1`
- **NLTK 句子切分数据**：`punkt_tab`（whisperx.align 依赖；约 11 MB；如 NLTK 下载服务器 SSL 异常会自动从 `~/nltk_data` 等系统路径回退拷贝）

这一步会持续 30–60 分钟（取决于带宽），总磁盘约 15–25 GB。可重复执行，已存在的会跳过。

子命令：
```bash
# 只下载部分 whisper 尺寸
uv run python deploy/scripts/download_models.py --whisper-sizes small large-v3

# 追加更多对齐语言
uv run python deploy/scripts/download_models.py --languages fa ka tr

# 上游字典里全部语言（约 40 种）
uv run python deploy/scripts/download_models.py --all-languages

# 跳过 diarization（如果暂不打算开启）
uv run python deploy/scripts/download_models.py --skip-diarization

# 同时下载 Silero VAD（默认走自带的 pyannote VAD）
uv run python deploy/scripts/download_models.py --include-silero
```

## 启动服务

```bash
uv run uvicorn deploy.api.server:app --host 0.0.0.0 --port 8000
```

启动日志会打印 `models/` 下当前已就绪的内容。Swagger UI: <http://localhost:8000/docs>。

## 冒烟测试

```bash
# 拉一段 11 秒的英文示例
curl -L -o /tmp/jfk.flac https://github.com/openai/whisper/raw/main/tests/jfk.flac

curl -s http://localhost:8000/health | jq
curl -s http://localhost:8000/models | jq

curl -s -X POST http://localhost:8000/transcribe \
  -F file=@/tmp/jfk.flac -F model=small -F language=en | jq '.segments[0]'
```

## 离线验证（推荐）

用 rootless network namespace（无需 sudo）冷启动一份独立服务，证明断网下完全能跑：

```bash
unshare -rn -- bash -c '
  ip link set lo up
  uv run uvicorn deploy.api.server:app --host 127.0.0.1 --port 8889 &
  PID=$!
  until curl -s --max-time 1 http://127.0.0.1:8889/health >/dev/null 2>&1; do sleep 2; done
  curl -s -X POST http://127.0.0.1:8889/transcribe \
    -F file=@/tmp/jfk.flac -F model=small -F language=en -F diarize=true | jq ".segments[0]"
  kill $PID
'
```

`unshare -rn` 创建了一个无默认路由的网络命名空间；如果端到端成功，且服务器日志里没有任何 `Downloading...` 行，就证明私有化部署完全自洽。

## 接口完整文档

参见 [docs/API.md](docs/API.md)。

## Docker 部署

仓库根有 [Dockerfile.cuda](../Dockerfile.cuda)、[Dockerfile.cpu](../Dockerfile.cpu)、[docker-compose.yml](../docker-compose.yml)、[.dockerignore](../.dockerignore)。

### 镜像变体

每个 Dockerfile 都用 multi-stage 构建，提供两个 target：

| target | 内容 | 大小（约） | 适用 |
|---|---|---|---|
| `base` | 代码 + Python 依赖，**不含模型** | CUDA 7 GB / CPU 3 GB | 开发；运行时 `-v $PWD/models:/app/models` 挂载 |
| `bundled`（默认） | base + COPY `./models` | CUDA 55 GB / CPU 50 GB | 离线分发，单镜像即可启动 |

构建前先确保 `./models/` 已通过 `download_models.py` 填好。

### GPU（CUDA）

```bash
# 构建 self-contained 镜像（默认 target）
docker build -f Dockerfile.cuda -t whisperx-private:cuda-bundled .

# 或：构建 slim base，运行时挂载 models/
docker build -f Dockerfile.cuda --target=base -t whisperx-private:cuda-base .

# 启动（self-contained）
docker run --rm --gpus all -p 8000:8000 whisperx-private:cuda-bundled

# 启动（mount models）
docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/models:/app/models:ro" \
  whisperx-private:cuda-base
```

### CPU

```bash
docker build -f Dockerfile.cpu -t whisperx-private:cpu-bundled .
docker run --rm -p 8000:8000 whisperx-private:cpu-bundled
```

> CPU 环境下推荐 `WHISPER_MODEL=tiny` 或 `base`；large 系列实际不可用。

### docker-compose

```bash
# GPU
docker compose --profile gpu build
docker compose --profile gpu up -d

# CPU
docker compose --profile cpu build
docker compose --profile cpu up -d

# 验证
curl http://localhost:8000/health
docker compose logs -f
```

可通过 `WHISPER_MODEL` / `LOG_LEVEL` 环境变量覆盖默认值，例如：

```bash
WHISPER_MODEL=large-v3 docker compose --profile gpu up -d
```

### 严格离线验证

bundled 镜像可在 `--network=none`（容器无任何网络）下正常工作：

```bash
docker run -d --name whisperx-iso --gpus all --network=none whisperx-private:cuda-bundled

# 等就绪
until docker exec whisperx-iso curl -fsS --max-time 1 http://127.0.0.1:8000/health >/dev/null 2>&1; do sleep 2; done

# 把音频拷进去测
docker cp /tmp/jfk.flac whisperx-iso:/tmp/jfk.flac
docker exec whisperx-iso curl -s -X POST http://127.0.0.1:8000/transcribe \
  -F file=@/tmp/jfk.flac -F model=small -F language=en -F diarize=true | jq

docker rm -f whisperx-iso
```

### 离线分发

镜像构建完成后，可导出为 tar 在内网机器上 import：

```bash
# 在能联网的机器上
docker save whisperx-private:cuda-bundled | zstd -T0 > whisperx-cuda.tar.zst

# 在内网机器上
zstd -d < whisperx-cuda.tar.zst | docker load
docker run --rm --gpus all -p 8000:8000 whisperx-private:cuda-bundled
```

bundled 镜像 ≈ 65 GB（CUDA 12.5 GB + 模型 50 GB）。压缩后大约 25–30 GB（已是 fp16/quantized，压缩比一般）。
