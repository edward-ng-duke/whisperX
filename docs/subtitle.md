# 视频字幕生成

用本机的 WhisperX API 直接给视频出字幕。视频不用先转音频，容器内有
ffmpeg，会自动抽音轨。

## 服务地址

| 场景 | URL |
|------|-----|
| 同机调用 | `http://localhost:8000` |
| 局域网 | `http://10.0.0.93:8000` |
| Tailscale | `http://100.124.71.66:8000` |

健康检查：`curl http://10.0.0.93:8000/health`

## 一、转写（视频 → JSON）

```bash
curl -X POST http://10.0.0.93:8000/transcribe \
  -F "file=@/path/to/video.mp4" \
  -F "model=large-v3" \
  -F "language=zh" \
  -F "diarize=true" \
  -o video.json
```

参数：

- `file` — mp4 / mkv / mov / webm / avi 都行
- `model` — `tiny` / `base` / `small` / `medium` / `large-v2` / `large-v3`
- `language` — `zh` / `en` / `ja` ...，省略则自动检测
- `diarize` — `true` 给每段标 speaker
- `num_speakers` / `min_speakers` / `max_speakers` — 已知人数时填，更准

GPU 上 large-v3 大约 0.1–0.3× 实时，10 分钟视频 1–3 分钟出结果。

## 二、JSON → SRT

API 只返回 JSON，自己转一下：

```python
# json_to_srt.py
import json, sys

def fmt(s):
    h = int(s) // 3600
    m = (int(s) // 60) % 60
    return f"{h:02d}:{m:02d}:{int(s)%60:02d},{int((s%1)*1000):03d}"

data = json.load(open(sys.argv[1]))
with open(sys.argv[2], "w") as f:
    for i, seg in enumerate(data["segments"], 1):
        spk = f"[{seg.get('speaker', '')}] " if seg.get("speaker") else ""
        f.write(f"{i}\n{fmt(seg['start'])} --> {fmt(seg['end'])}\n{spk}{seg['text'].strip()}\n\n")
```

```bash
python json_to_srt.py video.json video.srt
```

## 三、播放

```bash
mpv video.mp4 --sub-file=video.srt
```

VLC、PotPlayer 等也都能直接挂同名 `.srt`。

## 时间戳精度

- segment 级（粗，秒级）— 来自 Whisper + VAD
- words[] 词级（细，毫秒级）— 来自 wav2vec2 强制对齐

字幕断句用 segment 已经够；想做卡拉 OK 逐字高亮就用 `words[]`。
