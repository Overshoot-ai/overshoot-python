# Overshoot Python SDK

> **Warning: Alpha Release**: The API may change in future versions.

Python SDK for [Overshoot](https://overshoot.ai) real-time video analysis API.

## Requirements

- Python >= 3.10
- `aiohttp >= 3.9, < 4`
- `livekit >= 1.0.0`
- **FFmpeg** and **ffprobe** on PATH ([download](https://ffmpeg.org/download.html)) — required for `FileSource`, `RTSPSource`, `HLSSource`, `RTMPSource`, and `CameraSource`

## Installation

```bash
pip install git+https://github.com/Overshoot-ai/overshoot-python.git
```

## Quick Start

> See the [`examples/`](examples/) directory for complete runnable scripts.

### Check Available Models

```python
import asyncio
import overshoot

async def main():
    models = await overshoot.get_models(api_key="ovs_...")
    ready = [m for m in models if m.ready]
    for m in ready:
        print(f"{m.model}: {m.status}")

asyncio.run(main())
```

### Camera

```python
import asyncio
import overshoot

async def main():
    # Pick a ready model
    models = await overshoot.get_models(api_key="ovs_...")
    model = next(m.model for m in models if m.ready)

    client = overshoot.Overshoot(api_key="ovs_...")

    stream = await client.streams.create(
        source=overshoot.CameraSource(),
        prompt="Describe what you see",
        model=model,
        on_result=lambda r: print(r.result),
    )

    await asyncio.sleep(30)
    await stream.close()
    await client.close()

asyncio.run(main())
```

### Video File

```python
stream = await client.streams.create(
    source=overshoot.FileSource(path="/path/to/video.mp4", loop=True),
    prompt="Detect all objects",
    model="Qwen/Qwen3.5-9B",
    on_result=lambda r: print(r.result),
)
```

### RTSP Camera

```python
stream = await client.streams.create(
    source=overshoot.RTSPSource(url="rtsp://user:pass@192.168.1.10/stream"),
    prompt="Alert if someone enters the room",
    model="Qwen/Qwen3.5-9B",
    on_result=lambda r: print(r.result),
)
```

Uses TCP transport for reliability.

### HLS Stream

```python
stream = await client.streams.create(
    source=overshoot.HLSSource(url="https://example.com/live.m3u8"),
    prompt="Describe the scene",
    model="Qwen/Qwen3.5-9B",
    on_result=lambda r: print(r.result),
)
```

### RTMP Stream

```python
stream = await client.streams.create(
    source=overshoot.RTMPSource(url="rtmp://example.com/live/stream"),
    prompt="Describe the scene",
    model="Qwen/Qwen3.5-9B",
    on_result=lambda r: print(r.result),
)
```

### Custom Frames (OpenCV, numpy, etc.)

Push frames from any pipeline — OpenCV, PIL, a robotics stack, etc.

```python
import cv2
import numpy as np
import overshoot

source = overshoot.FrameSource(width=640, height=480)
stream = await client.streams.create(
    source=source,
    prompt="Count the people",
    model="Qwen/Qwen3.5-9B",
    on_result=lambda r: print(r.result),
)

cap = cv2.VideoCapture(0)
while True:
    ret, bgr = cap.read()
    if not ret:
        break
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
    rgba = cv2.resize(rgba, (640, 480))
    source.push_frame(rgba)  # accepts numpy arrays or raw bytes
```

### LiveKit (User-Managed)

If you manage your own LiveKit room:

```python
stream = await client.streams.create(
    source=overshoot.LiveKitSource(
        url="wss://your-livekit-server.com",
        token="your-livekit-token",
    ),
    prompt="Describe what you see",
    model="Qwen/Qwen3.5-9B",
    on_result=lambda r: print(r.result),
)
```

## Video Sources

| Source | Input | Requires FFmpeg |
|--------|-------|:-:|
| `FileSource(path, loop=False)` | Local video file | Yes |
| `RTSPSource(url)` | RTSP camera/server | Yes |
| `HLSSource(url)` | HLS live stream | Yes |
| `RTMPSource(url)` | RTMP stream | Yes |
| `CameraSource(device, width, height)` | Local camera | Yes |
| `FrameSource(width, height)` | Programmatic (push frames) | No |
| `LiveKitSource(url, token)` | User-managed LiveKit room | No |

## Client

### High-Level Client (Overshoot)

The `Overshoot` class manages streams with automatic keepalive, WebSocket result delivery, and resource cleanup.

```python
client = overshoot.Overshoot(
    api_key="ovs_...",
    base_url="https://api.overshoot.ai/v0.2",  # default
    timeout=30.0,
)
```

Always close when done:

```python
await client.close()
```

### Low-Level Client (ApiClient)

For direct HTTP control without background tasks or media pipelines:

```python
api = overshoot.ApiClient(api_key="ovs_...")

# Create stream (only accepts LiveKitSource or None for native transport)
response = await api.create_stream(
    source=overshoot.LiveKitSource(url="wss://...", token="..."),
    processing=overshoot.ClipProcessingConfig(target_fps=6),
    inference=overshoot.InferenceConfig(
        prompt="Describe what you see",
        model="Qwen/Qwen3.5-9B",
    ),
    mode="clip",
)

# Manage stream lifecycle manually
keepalive = await api.keepalive(response.stream_id)
config = await api.update_prompt(response.stream_id, "New prompt")
status = await api.close_stream(response.stream_id)

# Utilities
models = await api.get_models()
health = await api.health_check()

await api.close()
```

## Configuration

### `streams.create()` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source` | `SourceConfig` | required | Video source |
| `prompt` | `str` | required | Analysis prompt |
| `model` | `str` | required | Model name (use `get_models()`) |
| `on_result` | `Callable` | required | Callback for each result |
| `on_error` | `Callable` | `None` | Callback for errors |
| `mode` | `"clip"` or `"frame"` | `"frame"` | Stream mode |
| `output_schema` | `dict` | `None` | JSON schema for structured output |
| `max_output_tokens` | `int` | `None` | Cap tokens per inference request |
| `target_fps` | `int` | `6` | Clip mode: target frame sampling rate (1-30) |
| `clip_length_seconds` | `float` | `0.5` | Clip mode: clip duration |
| `delay_seconds` | `float` | `0.5` | Clip mode: delay between clips |
| `interval_seconds` | `float` | `0.5` | Frame mode: capture interval |

### Models

The list of available models is dynamic. Always use `get_models()` at runtime:

```python
models = await overshoot.get_models(api_key="ovs_...")
ready_models = [m for m in models if m.ready]
```

| Status | `ready` | Meaning |
|---|---|---|
| `"ready"` | `True` | Healthy, performing well |
| `"degraded"` | `True` | Near capacity, expect higher latency |
| `"saturated"` | `False` | At capacity, will reject new streams |
| `"unavailable"` | `False` | Endpoint not reachable |

### Processing Modes

#### Clip Mode (Default)

Processes short video clips with multiple frames — ideal for motion and temporal analysis.

```python
stream = await client.streams.create(
    source=source,
    prompt="What is happening?",
    model="Qwen/Qwen3.5-9B",
    on_result=handle_result,
    mode="clip",
    target_fps=6,
    clip_length_seconds=0.5,
    delay_seconds=0.5,
)
```

#### Frame Mode

Processes individual frames at regular intervals — ideal for static analysis, OCR.

```python
stream = await client.streams.create(
    source=source,
    prompt="Read all visible text",
    model="Qwen/Qwen3.5-9B",
    on_result=handle_result,
    mode="frame",
    interval_seconds=0.5,
)
```

### `max_output_tokens`

Caps the maximum tokens per inference request. The server enforces **128 effective output tokens per second** per stream:

```
effective_tokens_per_second = max_output_tokens * requests_per_second
```

If omitted, the server auto-calculates: `floor(128 * interval)`.

## Stream Object

```python
stream.stream_id   # server-assigned ID
stream.is_active   # True if running
```

### Update Prompt

```python
config = await stream.update_prompt("Now count the people")
```

### Close

```python
await stream.close()
```

## Structured Output

```python
schema = {
    "type": "object",
    "properties": {
        "people_count": {"type": "integer"},
        "description": {"type": "string"},
    },
    "required": ["people_count", "description"],
}

def handle_result(result: overshoot.StreamInferenceResult):
    data = result.result_json()
    print(f"People: {data['people_count']}")

stream = await client.streams.create(
    source=source,
    prompt="Count people and describe the scene",
    model="Qwen/Qwen3.5-9B",
    on_result=handle_result,
    output_schema=schema,
)
```

## Result Format

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Result ID |
| `stream_id` | `str` | Stream ID |
| `mode` | `"clip"` or `"frame"` | Processing mode |
| `model_name` | `str` | Model used |
| `prompt` | `str` | Task that was run |
| `result` | `str` | Model output |
| `inference_latency_ms` | `float` | Model inference time |
| `total_latency_ms` | `float` | End-to-end latency |
| `ok` | `bool` | Success status |
| `error` | `str \| None` | Error message if failed |
| `finish_reason` | `str \| None` | Why the model stopped (`"stop"`, `"length"`, `"content_filter"`) |

## Error Handling

```python
def handle_error(error: Exception):
    print(f"Stream error: {error}")

stream = await client.streams.create(
    source=source,
    prompt="Describe what you see",
    model="Qwen/Qwen3.5-9B",
    on_result=handle_result,
    on_error=handle_error,
)
```

| Exception | HTTP Status | Description |
|---|---|---|
| `AuthenticationError` | 401 | Invalid or revoked API key |
| `InsufficientCreditsError` | 402 | Not enough credits |
| `ValidationError` | 400/422 | Invalid request parameters |
| `NotFoundError` | 404 | Stream or resource not found |
| `ServerError` | 5xx | Server-side error |
| `NetworkError` | — | Connection/timeout failure |
| `StreamClosedError` | — | Operation on a closed stream |
| `WebSocketError` | — | WebSocket connection/protocol error |

## Limits & Billing

- **Concurrent streams:** Maximum 5 per API key (429 error if exceeded).
- **Output token rate:** 128 effective tokens per second per stream.
- **Billing:** By stream duration, not inference count.

## Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("overshoot").setLevel(logging.DEBUG)
```

## License

MIT
