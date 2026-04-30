# Speaker Diarization 调参指南

本文整理 WhisperX 私有化部署里 **说话人分离 (Speaker Diarization)** 全链路上**所有可调的参数**，分三层：

1. `POST /diarize` (以及 `POST /transcribe?diarize=true`) HTTP 端点暴露的入参 — 直接传，最常用
2. pyannote-community-1 模型本身的内部超参 — 写在 `config.yaml`，要在加载时 `pipeline.instantiate({...})` 覆盖
3. 我们自己的后处理（`qwen3_diarize.py` 里的 turn 合并） — 改 Python 默认值或加 CLI 参数

底层模型链路是 `whisperx.diarize.DiarizationPipeline` → `pyannote.audio.Pipeline`（model = `pyannote/speaker-diarization-community-1`）。

---

## 1. 三层概览

| 层 | 在哪里 | 作用域 | 何时调 |
|---|---|---|---|
| ① HTTP 入参 | `POST /diarize` form 字段 | 单次请求 | 已知人数 / 想约束聚类范围 |
| ② 模型超参 | `config.yaml` + `Pipeline.instantiate()` | 进程内全局，重启失效 | 想改聚类敏感度、合并短停顿 |
| ③ 后处理 | `deploy/scripts/qwen3_diarize.py` 的 `merge_turns()` | 仅影响 qwen3+diarize 输出粒度 | 想要更粗 / 更细的 turn |

---

## 2. 第一层：HTTP 端点参数

`POST /diarize` 与 `POST /transcribe?diarize=true` 都接受同样三个 form 字段，最终透传给 pyannote `SpeakerDiarization.__call__`。

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `num_speakers` | int | `null` | **精确**说话人数。给定后会硬性指定簇数。 |
| `min_speakers` | int | `null` | 聚类下界。 |
| `max_speakers` | int | `null` | 聚类上界。 |

三个都不传 = 完全自动；优先级 `num_speakers` > `min/max_speakers`。

**curl 示例**

```bash
# 自动 (默认)
curl -X POST http://localhost:8000/diarize \
  -F "file=@meeting.wav"

# 已知是 2 人对话
curl -X POST http://localhost:8000/diarize \
  -F "file=@meeting.wav" \
  -F "num_speakers=2"

# 至少 3 人，最多 5 人 (会议)
curl -X POST http://localhost:8000/diarize \
  -F "file=@meeting.wav" \
  -F "min_speakers=3" \
  -F "max_speakers=5"
```

**何时用每一项**

- 知道**确切**人数 → `num_speakers`，最稳。
- 知道**上界**（"最多 4 个人"）→ 只传 `max_speakers`，避免被误聚成更多 speaker。
- 知道**下界**（"至少 2 个人"）→ 只传 `min_speakers`，避免一人独占。
- 完全未知 → 都不传。但**短音频 (<60s) 自动判别可能偏保守**，可能漏掉只说一两次话的人。

**代码路径**

- 端点定义: [`deploy/api/server.py`](../deploy/api/server.py) 的 `diarize_endpoint`
- 业务封装: [`deploy/api/pipeline.py`](../deploy/api/pipeline.py) 的 `diarize_only()`
- 上游调用: [`whisperx/diarize.py`](../whisperx/diarize.py) 的 `DiarizationPipeline.__call__`

---

## 3. 第二层：pyannote-community-1 模型超参

模型默认配置文件位于：

```
models/hf/hub/models--pyannote--speaker-diarization-community-1/snapshots/<rev>/config.yaml
```

当前内容：

```yaml
pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: VBxClustering
    segmentation: $model/segmentation
    segmentation_batch_size: 32
    embedding: $model/embedding
    embedding_batch_size: 32
    embedding_exclude_overlap: true
    plda: $model/plda

params:
  clustering:
    threshold: 0.6
    Fa: 0.07
    Fb: 0.8
  segmentation:
    min_duration_off: 0.0
```

可在加载完 pipeline 后用 `pipe.instantiate({...})` 在运行时覆盖（对所有后续请求生效，进程重启失效）。

### 3.1 核心：`clustering.threshold`

VBx 聚类的距离阈值。**最值得先动的一个**。

| 调整方向 | 效果 |
|---|---|
| **调高** (0.6 → 0.7 / 0.8) | 距离更宽容，**更难**形成新簇，倾向**少 speaker**。同一人被切成两个 speaker 时调这个。 |
| **调低** (0.6 → 0.5 / 0.4) | 容易开新簇，倾向**多 speaker**。两个相似声音被混成一人时调这个。 |

### 3.2 `clustering.Fa` / `clustering.Fb`

VBx 的内部声学权重。模型作者已调好，**不建议轻动**。除非要做大量数据上的 grid search 找最优。

### 3.3 `segmentation.min_duration_off`

短于此值的**静音间隙**会被吸收，把两段语音合成一段。默认 `0.0` = 不吸收。

| 值 | 效果 |
|---|---|
| `0.0` | 任何停顿都算切分点。短停顿密集时 turn 极碎。 |
| `0.3 – 0.5` | 吃掉换气类停顿，一句话不会被打断。**会议/对话推荐。** |
| `1.0+` | 把整段长发言粘在一起，可能把不同 turn 错并。 |

### 3.4 性能向：`segmentation_batch_size` / `embedding_batch_size`

只影响推理吞吐，不影响结果。显存紧时调小，显存富裕时调大（默认 32 已经很激进）。

### 3.5 如何覆盖

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", ...)

pipe.instantiate({
    "clustering":   {"threshold": 0.7},
    "segmentation": {"min_duration_off": 0.5},
})

# 之后 pipe(audio) 全部用新阈值
```

我们的封装在 [`deploy/api/pipeline.py`](../deploy/api/pipeline.py) 的 `get_diarize_pipeline()` 里**不调** `instantiate`，所以走的是 `config.yaml` 里写的默认值。要改默认行为，可以：

- **临时**：在 Python REPL/脚本里拿到 `pipeline._diarize_pipeline` 调 `.instantiate(...)`
- **持久**：在 `get_diarize_pipeline()` 里 `Pipeline.from_pretrained(...)` 之后立刻 `.instantiate(...)`，或直接编辑 `config.yaml`（注意会被模型重新下载覆盖）

> 当前 `/diarize` 端点**没有暴露**这一层超参。如果以后想在 HTTP 上直接传 `clustering_threshold` / `min_duration_off`，需要改 `diarize_only()` 签名 + endpoint。

---

## 4. 第三层：后处理 `merge_turns()`

我们在 [`deploy/scripts/qwen3_diarize.py`](../deploy/scripts/qwen3_diarize.py) 里有一个轻量后处理，把同 speaker 相邻段合并成一个 turn，再喂给 ASR：

```python
def merge_turns(segments, gap_tol: float = 1.0):
    """同 speaker 相邻段，间隙 ≤ gap_tol 自动合并成一个 turn。"""
```

| `gap_tol` | 效果 |
|---|---|
| `0.0` | 完全不合并，turn 数 = pyannote 原始段数（最碎） |
| `1.0` (当前默认) | 1s 内的同 speaker 停顿吸收，适合连贯口语 |
| `2.0 – 3.0` | 长发言（带换气、长停顿）粘合成一个 turn，turn 更粗 |

> 这一层**只影响** `qwen3_diarize.py` 输出的 turn 粒度（决定 ASR 的切片边界），**不影响** `/diarize` 接口本身的返回。如果你只调 `/diarize`，这个旋钮无关。

附近还有一个 `ASR_MIN_SEC = 0.5`（同文件顶部），低于这个时长的 turn 不送 ASR。如果想抓 backchannel ("嗯/啊/对")，把它降到 0.2 左右；当然 Qwen3 自己有 0.5s 下限，再低就报错。

---

## 5. 场景调参速查

按音频类型给推荐组合，左到右：HTTP 入参 → 模型超参 → 后处理 `gap_tol`。

| 场景 | num/min/max | clustering.threshold | min_duration_off | gap_tol |
|---|---|---|---|---|
| 单人演讲 / 独白 | — | 默认 0.6 | 默认 0.0 | 默认 1.0 |
| 双人对话 (访谈、采访) | `max=2` | 0.65 | 0.5 | 1.5 |
| 小型会议 (3–5 人) | `min=3 max=6` | 默认 0.6 | 0.5 | 1.0 |
| 圆桌 / 多人 (>5) | `min=5` | 0.55 | 0.3 | 0.5 |
| 电话客服 (短交替) | `num=2` | 0.55 | 0.0 | 0.0 |
| 嘈杂环境 / 远场 | 视情况 | 0.7 | 0.5 | 1.5 |

> 经验值，仅作起点；在自己语料上 A/B 几次再固化。

---

## 6. 故障排查

| 症状 | 可能原因 | 调什么 |
|---|---|---|
| **同一人被切成多个 speaker** (`SPEAKER_00 / SPEAKER_02 / SPEAKER_03` 实际是同一人) | `threshold` 偏低；或音频里这个人音色变化大（情绪、距麦距离） | `clustering.threshold` ↑ (0.6 → 0.7 / 0.75)；或显式 `max_speakers` |
| **两个相似声音被并成一人** (两个男生只检出 1 个 speaker) | `threshold` 偏高 | `clustering.threshold` ↓ (0.6 → 0.5)；或显式 `min_speakers` |
| **一句完整话被切成多 turn** | `min_duration_off=0` 不吸收停顿 + 后处理 `gap_tol` 不够 | `min_duration_off` ↑ (0.0 → 0.5) **或** `gap_tol` ↑ (1.0 → 2.0) |
| **backchannel ("嗯/啊/对") 丢失** | 时长 < 0.5s 被 `ASR_MIN_SEC` 跳过 | 把 `ASR_MIN_SEC` 降到 0.3 (Qwen3 仍要 ≥0.5，再低会 ASR 失败但 diarize 段保留) |
| **`num_speakers=2` 还出现 `SPEAKER_03`** | 上游 pyannote 在 segment-level 仍可能多分；最终 cluster 数受 `num_speakers` 约束。复查日志或检查实际返回 | 一般忽略；或 `num_speakers` 改成 `max_speakers=2 min_speakers=2` |
| **结果不稳定，每次跑略不同** | pyannote 内含随机性 (聚类初始化) | 在 `Pipeline.from_pretrained()` 后设 `torch.manual_seed(...)`；或可接受小幅波动 |

---

## 7. 附录：当前默认值速查表

| 参数 | 默认 | 在哪 |
|---|---|---|
| (HTTP) `num_speakers` | `null` | [`deploy/api/server.py`](../deploy/api/server.py) `diarize_endpoint` |
| (HTTP) `min_speakers` | `null` | 同上 |
| (HTTP) `max_speakers` | `null` | 同上 |
| `clustering.threshold` | `0.6` | `models/hf/.../config.yaml` |
| `clustering.Fa` | `0.07` | 同上 |
| `clustering.Fb` | `0.8` | 同上 |
| `segmentation.min_duration_off` | `0.0` | 同上 |
| `segmentation_batch_size` | `32` | 同上 |
| `embedding_batch_size` | `32` | 同上 |
| `embedding_exclude_overlap` | `true` | 同上 |
| (post) `gap_tol` | `1.0` | [`deploy/scripts/qwen3_diarize.py`](../deploy/scripts/qwen3_diarize.py) `merge_turns` |
| (post) `ASR_MIN_SEC` | `0.5` | 同文件顶部 |
| (post) `ASR_MAX_SEC` | `300.0` | 同上（Qwen3 服务端硬限） |

---

## 8. 延伸阅读

- pyannote.audio 文档：<https://github.com/pyannote/pyannote-audio>
- VBx 聚类原文：Diez et al. *"Bayesian HMM clustering of x-vector sequences (VBx)"*, 2020
- 模型卡: <https://huggingface.co/pyannote/speaker-diarization-community-1>
- API 文档：[API.md](API.md)
