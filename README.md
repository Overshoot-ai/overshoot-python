# Overshoot Python SDK

> **Warning: Alpha Release**: The API may change in future versions.

Python SDK for [Overshoot](https://overshoot.ai) real-time video analysis API.

## Installation

```bash
pip install overshoot
```

## Quick Start

> **Note:** The `base_url` parameter is optional and defaults to `https://api.overshoot.ai/v0.2`.
> You can omit it for standard usage or provide a custom URL for private deployments.

### Camera Source

```python
import asyncio
import overshoot

async def main():
    client = overshoot.Overshoot(api_key="sk-...")

    stream = await client.streams.create(
        source=overshoot.CameraSource(),
        prompt="Describe what you see",
        on_result=lambda r: print(r.result),
    )

    await asyncio.sleep(30)
    await stream.close()
    await client.close()

asyncio.run(main())
```

### Video File Source

```python
stream = await client.streams.create(
    source=overshoot.FileSource(path="/path/to/video.mp4", loop=True),
    prompt="Detect all objects in the video and count them",
    on_result=lambda r: print(r.result),
)
```

> **Note:** Set `loop=True` for video files to loop continuously until you call `stream.close()`.

### LiveKit Source

If you're on a restrictive network where direct WebRTC connections fail, you can use LiveKit as an alternative video transport.

```python
stream = await client.streams.create(
    source=overshoot.LiveKitSource(
        url="wss://your-livekit-server.com",
        token="your-livekit-token",
    ),
    prompt="Describe what you see",
    on_result=lambda r: print(r.result),
)
```

> **Note:** With a LiveKit source, you are responsible for publishing video to the LiveKit room using the [LiveKit Python SDK](https://github.com/livekit/python-sdks).

### WebRTC Source

Provide a raw SDP offer from your own WebRTC peer connection.

```python
source = overshoot.WebRTCSource(sdp="v=0\r\n...")
```

## Client

The `Overshoot` class is the main entry point. It manages an HTTP session and provides the `streams` namespace for creating and managing streams.

```python
client = overshoot.Overshoot(
    api_key="sk-...",
    base_url="https://api.overshoot.ai/v0.2",  # default
    timeout=30.0,  # HTTP timeout in seconds
)
```

Always close the client when done:

```python
await client.close()
```

## Configuration

### `streams.create()` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source` | `SourceConfig` | required | Video source |
| `prompt` | `str` | required | Analysis prompt |
| `on_result` | `Callable` | required | Callback for each result |
| `on_error` | `Callable` | `None` | Callback for errors |
| `mode` | `"clip"` or `"frame"` | auto | Stream mode |
| `backend` | `"overshoot"` or `"gemini"` | `"overshoot"` | Model backend |
| `model` | `str` | `"Qwen/Qwen3-VL-30B-A3B-Instruct"` | Model name |
| `output_schema` | `dict` | `None` | JSON schema for structured output |
| `max_output_tokens` | `int` | `None` | Cap tokens per inference request |
| `sampling_ratio` | `float` | `1.0` | Clip mode: frame sampling ratio |
| `fps` | `int` | `30` | Clip mode: frames per second |
| `clip_length_seconds` | `float` | `0.2` | Clip mode: clip duration |
| `delay_seconds` | `float` | `0.2` | Clip mode: delay between clips |
| `interval_seconds` | `float` | `0.2` | Frame mode: capture interval |

### `max_output_tokens`

Caps the maximum number of tokens the model can generate per inference request. To ensure low latency, the server enforces a limit of **128 effective output tokens per second** per stream:

```
effective_tokens_per_second = max_output_tokens × requests_per_second
```

Where `requests_per_second` is `1 / delay_seconds` (clip mode) or `1 / interval_seconds` (frame mode).

**If omitted**, the server auto-calculates the optimal value: `floor(128 × interval)`. For example, with `delay_seconds=0.5` (2 requests/sec), the server defaults to `floor(128 × 0.5) = 64` tokens per request.

**If provided**, the server validates that `max_output_tokens / interval ≤ 128`. If exceeded, the request is rejected with a 422 error.

| Scenario | Interval | Requests/sec | Max allowed `max_output_tokens` |
|---|---|---|---|
| Clip mode, fast updates | 0.2s | 5 | 25 |
| Clip mode, default | 0.5s | 2 | 64 |
| Clip mode, slow | 1.0s | 1 | 128 |
| Frame mode, default | 0.2s | 5 | 25 |
| Frame mode, slow | 2.0s | 0.5 | 256 |
| Frame mode, very slow | 5.0s | 0.2 | 640 |

```python
stream = await client.streams.create(
    source=source,
    prompt="Describe what you see briefly",
    on_result=handle_result,
    max_output_tokens=100,    # Must satisfy: 100 / delay_seconds <= 128
    delay_seconds=1.0,        # 1 request/sec -> 100 tokens/sec <= 128
)
```

### Models

The list of available models is dynamic and changes over time. Always use `get_models()` to fetch the current list and check which models are ready before starting a stream.

```python
api = overshoot.ApiClient(api_key="sk-...")
models = await api.get_models()

# Only use models that are ready
ready_models = [m for m in models if m.ready]
for model in ready_models:
    print(f"{model.model}: {model.status}")

await api.close()
```

At the time of writing, available models include:

| Model | Size | Notes |
|---|---|---|
| `Qwen/Qwen3-VL-2B-Instruct` | 2B | Fastest, lightweight tasks |
| `Qwen/Qwen3-VL-4B-Instruct` | 4B | Good balance of speed and quality |
| `Qwen/Qwen3-VL-8B-Instruct` | 8B | Particularly good at OCR and text extraction |
| `Qwen/Qwen3-VL-30B-A3B-Instruct` | 30B (MoE) | Very fast general-purpose model (default) |
| `Qwen/Qwen3-VL-32B-Instruct-FP8` | 32B | Higher quality, FP8 quantized |
| `OpenGVLab/InternVL3_5-4B` | 4B | Good at visual detail, lightweight |
| `OpenGVLab/InternVL3_5-30B-A3B` | 30B (MoE) | Excels at visual detail, more verbose |
| `openbmb/MiniCPM-V-4_5` | — | Strong multimodal understanding |
| `allenai/Molmo2-8B` | 8B | Research-grade vision-language model |

> **Note:** This list is a snapshot. Models may be added, removed, or change status at any time. Always call `get_models()` at runtime to get the current state.

Each model has a `status` indicating its current availability:

| Status | `ready` | Meaning | Action |
|---|---|---|---|
| `"ready"` | `True` | Healthy, performing well | Use this model |
| `"degraded"` | `True` | Near capacity, expect higher latency | Usable, but consider alternatives |
| `"saturated"` | `False` | At capacity, will reject new streams | Pick a different model |
| `"unavailable"` | `False` | Endpoint not reachable | Pick a different model |

### Processing Modes

The SDK supports two processing modes:

#### Clip Mode (Default)

Processes short video clips with multiple frames, ideal for motion analysis and temporal understanding.

```python
stream = await client.streams.create(
    source=source,
    prompt="What is happening in this scene?",
    on_result=handle_result,
    mode="clip",
    sampling_ratio=1.0,          # Process 100% of frames (default)
    clip_length_seconds=0.2,     # 0.2 second clips (default)
    delay_seconds=0.2,           # New clip every 0.2s (default)
    fps=30,                      # Frames per second (default: 30)
)
```

**Use cases:** Activity detection, gesture recognition, motion tracking, sports analysis

#### Frame Mode

Processes individual frames at regular intervals, ideal for static analysis and fast updates.

```python
stream = await client.streams.create(
    source=source,
    prompt="Describe this frame",
    on_result=handle_result,
    mode="frame",
    interval_seconds=0.2,  # Capture frame every 0.2s (default)
)
```

**Use cases:** OCR, object detection, scene description, static monitoring

> **Note:** If you don't specify a mode, the SDK defaults to clip mode. Mode is automatically inferred if you only pass `interval_seconds`.

### Processing Parameters Explained

#### Clip Mode Parameters

- **`fps`**: The frame rate of your video source (default: 30).
- **`sampling_ratio`**: What fraction of frames to include in each clip (1.0 = 100% of frames, default).
- **`clip_length_seconds`**: Duration of video captured for each inference (default: 0.2 seconds).
- **`delay_seconds`**: How often inference runs (default: 0.2 seconds — 5 inferences per second).

**Example with defaults:** `fps=30`, `clip_length_seconds=0.2`, `sampling_ratio=1.0`, `delay_seconds=0.2`:

- Each clip captures 0.2 seconds of video (6 frames at 30fps)
- 100% of frames are sampled = 6 frames sent to the model
- New clip starts every 0.2 seconds = ~5 inference results per second

#### Frame Mode Parameters

- **`interval_seconds`**: Time between frame captures (default: 0.2 seconds — 5 frames per second).

#### Configuration by Use Case

**Real-time tracking** (low latency, frequent updates) — Clip Mode:

```python
stream = await client.streams.create(
    source=source,
    prompt="Track the person",
    on_result=handle_result,
    sampling_ratio=1.0,
    clip_length_seconds=0.2,
    delay_seconds=0.2,
)
```

**Event detection** (monitoring for specific occurrences) — Clip Mode:

```python
stream = await client.streams.create(
    source=source,
    prompt="Alert if someone enters the room",
    on_result=handle_result,
    sampling_ratio=0.5,
    clip_length_seconds=3.0,
    delay_seconds=2.0,
)
```

**Fast OCR/Detection** (static analysis) — Frame Mode:

```python
stream = await client.streams.create(
    source=source,
    prompt="Read all visible text",
    on_result=handle_result,
    mode="frame",
    interval_seconds=0.5,
)
```

## Stream Object

`client.streams.create()` returns a `Stream`. It runs background tasks for WebSocket result delivery and lease keepalive.

```python
stream.stream_id   # server-assigned ID
stream.is_active   # True if the stream is running
```

### Update Prompt

Change the analysis prompt while the stream is running:

```python
config = await stream.update_prompt("Now count the people in the scene")
print(config.prompt)  # updated prompt
```

### Close

Stop the stream, clean up resources, and notify the server:

```python
await stream.close()
```

Closing is idempotent. The stream automatically renews its lease in the background. If keepalive fails, the stream closes itself and fires `on_error`.

## Structured Output

Pass a JSON schema to get structured responses:

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
    print(f"People: {data['people_count']}, Description: {data['description']}")

stream = await client.streams.create(
    source=source,
    prompt="Count people and describe the scene",
    on_result=handle_result,
    output_schema=schema,
)
```

The model will return valid JSON matching your schema. If the model cannot produce valid output, `result.ok` will be `False` and `result.error` will contain details.

> **Note:** `result.result` is always a string. When using `output_schema`, parse it with `result.result_json()` or `json.loads(result.result)`.

## Result Format

The `on_result` callback receives a `StreamInferenceResult`:

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Result ID |
| `stream_id` | `str` | Stream ID |
| `mode` | `"clip"` or `"frame"` | Processing mode used |
| `model_backend` | `str` | Model backend |
| `model_name` | `str` | Model used |
| `prompt` | `str` | Task that was run |
| `result` | `str` | Model output (always a string — parse JSON if using `output_schema`) |
| `inference_latency_ms` | `float` | Model inference time |
| `total_latency_ms` | `float` | End-to-end latency |
| `ok` | `bool` | Success status |
| `error` | `str` or `None` | Error message if failed |
| `finish_reason` | `str` or `None` | Why the model stopped generating |

The `finish_reason` field indicates why the model stopped generating:

- `"stop"` — Model finished naturally
- `"length"` — Output was truncated because it hit `max_output_tokens`. Consider increasing the value or using a longer processing interval.
- `"content_filter"` — Output was blocked by safety filtering

## Stream Lifecycle

### Keepalive

Streams have a server-side lease (30 second TTL). The SDK automatically sends keepalive requests to renew it. You don't need to manage keepalives manually.

If a keepalive fails (e.g., network issues), the stream will stop and `on_error` will be called. If your account runs out of credits, the keepalive returns a 402 error and the stream expires — this is terminal and requires starting a new stream after adding credits.

**Network disconnects are permanent.** If the client loses connectivity for more than 30 seconds, the lease expires and the stream is destroyed. There is no automatic reconnection — you must call `streams.create()` to create a new stream.

### State and Memory

The SDK does not maintain memory or state between inference calls — each frame/clip is processed independently. If your application needs to track state over time (e.g., counting repetitions, detecting transitions), implement this in your `on_result` callback:

```python
last_position = "up"
rep_count = 0

def handle_result(result: overshoot.StreamInferenceResult):
    global last_position, rep_count
    data = result.result_json()

    if last_position == "down" and data["position"] == "up":
        rep_count += 1
        print(f"Rep count: {rep_count}")
    last_position = data["position"]
```

## Prompt Engineering

Prompt quality significantly affects results. Here are some tips:

**Be specific about output format:**

```python
prompt = "Count the people visible. Return only a number."
```

**Include examples for complex tasks:**

```python
prompt = """Describe the primary action happening. Examples:
- "Person walking left"
- "Car turning right"
- "Dog sitting still"
"""
```

**Request minimal output for lower latency:**

```python
prompt = "Is there a person in frame? Answer only 'yes' or 'no'."
```

**Provide context when needed:**

```python
prompt = f"You are monitoring a {location_name}. Alert if you see: {', '.join(alert_conditions)}."
```

**Use JSON schema for structured data** (see [Structured Output](#structured-output)).

## Error Handling

### Callback Errors

Pass `on_error` to handle runtime errors (keepalive failures, WebSocket issues):

```python
def handle_error(error: Exception):
    print(f"Stream error: {error}")

stream = await client.streams.create(
    source=source,
    prompt="Describe what you see",
    on_result=handle_result,
    on_error=handle_error,
)
```

### Exception Hierarchy

All exceptions inherit from `OvershootError`:

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

```python
try:
    stream = await client.streams.create(...)
except overshoot.AuthenticationError:
    print("Bad API key")
except overshoot.InsufficientCreditsError:
    print("Out of credits")
except overshoot.ApiError as e:
    print(f"API error {e.status_code}: {e}")
```

`ApiError` exposes `status_code`, `request_id`, and `details` attributes.

## Low-Level API Client

For direct HTTP control without background tasks or WebSocket management, use `ApiClient`:

```python
api = overshoot.ApiClient(api_key="sk-...")

# Create a stream
response = await api.create_stream(
    source=overshoot.LiveKitSource(url="wss://...", token="..."),
    processing=overshoot.ClipProcessingConfig(
        sampling_ratio=1.0,
        fps=30,
        clip_length_seconds=0.2,
        delay_seconds=0.2,
    ),
    inference=overshoot.InferenceConfig(
        prompt="Describe what you see",
        max_output_tokens=25,  # Optional: must satisfy max_output_tokens / delay_seconds <= 128
    ),
    mode="clip",
)
print(response.stream_id)

# Renew lease
keepalive = await api.keepalive(response.stream_id)

# Update prompt
config = await api.update_prompt(response.stream_id, "New prompt")

# Close stream
status = await api.close_stream(response.stream_id)

# List available models
models = await api.get_models()

# Health check
health = await api.health_check()

# Submit feedback
await api.submit_feedback(response.stream_id, rating=5, category="quality", feedback="Great results")

# Get all feedback
feedback = await api.get_feedback()

await api.close()
```

`ApiClient` only accepts wire-ready sources (`LiveKitSource`, `WebRTCSource`). If you need `FileSource` or `CameraSource`, use the high-level `Overshoot` client which handles WebRTC negotiation automatically.

## Use Cases

- Real-time text extraction and OCR
- Safety monitoring (PPE detection, hazard identification)
- Accessibility tools (scene description)
- Gesture recognition and control
- Document scanning and alignment detection
- Sports and fitness form analysis
- Video file content analysis
- Screen content monitoring and analysis
- Tutorial and training content analysis
- Application monitoring and testing

## Limits & Billing

- **Concurrent streams:** Maximum 5 streams per API key. Attempting to create a 6th stream returns a 429 error. Close existing streams with `stream.close()` before starting new ones.
- **Output token rate:** 128 effective tokens per second per stream (see [`max_output_tokens`](#max_output_tokens)).
- **Billing:** Streams are billed by duration (stream time), not by number of inference requests. A stream running for 60 seconds costs the same regardless of processing interval. Billing starts at stream creation and ends when the stream is closed or expires.

## Logging

The SDK uses Python's standard `logging` module under the `"overshoot"` logger.

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("overshoot").setLevel(logging.DEBUG)
```

## Requirements

- Python >= 3.10
- `aiohttp >= 3.9, < 4`
- `aiortc >= 1.9.0`

## Feedback

As this is an alpha release, we welcome your feedback! Please report issues or suggestions through GitHub issues.

## License

MIT
