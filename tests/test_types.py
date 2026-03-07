"""Tests for overshoot.types — dataclass construction and helpers."""
import json

from overshoot import (
    CameraSource,
    ClipProcessingConfig,
    FileSource,
    FrameSource,
    FrameProcessingConfig,
    HLSSource,
    InferenceConfig,
    KeepaliveResponse,
    Lease,
    LiveKitSource,
    ModelInfo,
    RTMPSource,
    RTSPSource,
    StatusResponse,
    StreamConfigResponse,
    StreamCreateResponse,
    StreamInferenceResult,
)


# ── Source types ──────────────────────────────────────────────────────


def test_livekit_source():
    s = LiveKitSource(url="wss://lk.example.com", token="tok-123")
    assert s.url == "wss://lk.example.com"
    assert s.token == "tok-123"


def test_file_source_defaults():
    s = FileSource(path="./video.mp4")
    assert s.path == "./video.mp4"
    assert s.loop is False


def test_file_source_loop():
    s = FileSource(path="./video.mp4", loop=True)
    assert s.loop is True


def test_rtsp_source():
    s = RTSPSource(url="rtsp://192.168.1.10/stream")
    assert s.url == "rtsp://192.168.1.10/stream"


def test_hls_source():
    s = HLSSource(url="https://example.com/live.m3u8")
    assert s.url == "https://example.com/live.m3u8"


def test_rtmp_source():
    s = RTMPSource(url="rtmp://example.com/live/stream")
    assert s.url == "rtmp://example.com/live/stream"


def test_camera_source_default():
    s = CameraSource()
    assert s.device == "default"
    assert s.width == 1280
    assert s.height == 720


def test_camera_source_custom():
    s = CameraSource(device="/dev/video1", width=1920, height=1080)
    assert s.device == "/dev/video1"
    assert s.width == 1920
    assert s.height == 1080


def test_frame_source():
    s = FrameSource(width=640, height=480)
    assert s.width == 640
    assert s.height == 480


def test_frame_source_push_before_connect_raises():
    s = FrameSource(width=640, height=480)
    try:
        s.push_frame(b"\x00" * (640 * 480 * 4))
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not connected" in str(e)


def test_frame_source_push_wrong_size_raises():
    """push_frame should raise ValueError for wrong-sized data."""
    s = FrameSource(width=2, height=2)
    # Simulate connected state
    from unittest.mock import MagicMock
    s._livekit_video_source = MagicMock()

    try:
        s.push_frame(b"\x00" * 10)  # wrong size (should be 2*2*4=16)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "mismatch" in str(e)


# ── Processing configs ───────────────────────────────────────────────


def test_clip_processing_config_target_fps():
    c = ClipProcessingConfig(target_fps=6)
    assert c.target_fps == 6
    assert c.clip_length_seconds == 0.5
    assert c.delay_seconds == 0.5


def test_frame_processing_config():
    f = FrameProcessingConfig(interval_seconds=3.0)
    assert f.interval_seconds == 3.0


# ── Inference config ─────────────────────────────────────────────────


def test_inference_config():
    i = InferenceConfig(prompt="Describe", model="Qwen/Qwen3.5-9B")
    assert i.prompt == "Describe"
    assert i.model == "Qwen/Qwen3.5-9B"
    assert i.output_schema_json is None
    assert i.max_output_tokens is None


def test_inference_config_max_output_tokens():
    i = InferenceConfig(prompt="Describe", model="test-model", max_output_tokens=512)
    assert i.max_output_tokens == 512


# ── Response types ───────────────────────────────────────────────────


def test_stream_create_response():
    r = StreamCreateResponse(
        stream_id="abc-123",
        lease=Lease(ttl_seconds=45),
    )
    assert r.stream_id == "abc-123"
    assert r.lease is not None
    assert r.lease.ttl_seconds == 45


def test_keepalive_response():
    k = KeepaliveResponse(status="ok", stream_id="abc", ttl_seconds=45, cost_cents=0.05)
    assert k.ttl_seconds == 45
    assert k.cost_cents == 0.05
    assert k.credits_remaining_cents is None
    assert k.seconds_charged is None


def test_keepalive_response_seconds_charged():
    k = KeepaliveResponse(
        status="ok", stream_id="abc", ttl_seconds=45, seconds_charged=5.2
    )
    assert k.seconds_charged == 5.2


def test_stream_config_response():
    c = StreamConfigResponse(
        id="cfg-1", stream_id="str-1", prompt="test", model="Qwen"
    )
    assert c.prompt == "test"


def test_status_response():
    s = StatusResponse(status="ok")
    assert s.status == "ok"


def test_stream_inference_result_ok():
    r = StreamInferenceResult(
        id="res-1",
        stream_id="str-1",
        mode="clip",
        model_name="Qwen",
        prompt="test",
        result="a car is visible",
        inference_latency_ms=150.0,
        total_latency_ms=200.0,
        ok=True,
    )
    assert r.ok is True
    assert r.error is None
    assert r.finish_reason is None
    assert r.result == "a car is visible"


def test_stream_inference_result_with_finish_reason():
    r = StreamInferenceResult(
        id="res-1",
        stream_id="str-1",
        mode="clip",
        model_name="Qwen",
        prompt="test",
        result="a car is visible",
        inference_latency_ms=150.0,
        total_latency_ms=200.0,
        ok=True,
        finish_reason="stop",
    )
    assert r.finish_reason == "stop"


def test_stream_inference_result_error():
    r = StreamInferenceResult(
        id="res-2",
        stream_id="str-1",
        mode="frame",
        model_name="test-model",
        prompt="test",
        result="",
        inference_latency_ms=0,
        total_latency_ms=0,
        ok=False,
        error="model timeout",
    )
    assert r.ok is False
    assert r.error == "model timeout"


def test_result_json():
    data = {"count": 3, "objects": ["car", "person", "dog"]}
    r = StreamInferenceResult(
        id="res-3",
        stream_id="str-1",
        mode="clip",
        model_name="Qwen",
        prompt="count objects",
        result=json.dumps(data),
        inference_latency_ms=100,
        total_latency_ms=150,
        ok=True,
    )
    parsed = r.result_json()
    assert parsed == data
    assert parsed["count"] == 3


def test_result_json_invalid():
    r = StreamInferenceResult(
        id="res-4",
        stream_id="str-1",
        mode="clip",
        model_name="Qwen",
        prompt="test",
        result="not valid json",
        inference_latency_ms=100,
        total_latency_ms=150,
        ok=True,
    )
    try:
        r.result_json()
        assert False, "Should have raised ValueError"
    except (ValueError, json.JSONDecodeError):
        pass


def test_frozen_dataclasses():
    """Frozen dataclasses should be immutable."""
    s = LiveKitSource(url="wss://test", token="tok")
    try:
        s.url = "changed"  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass


def test_model_info():
    m = ModelInfo(model="Qwen/Qwen3.5-9B", ready=True, status="ready")
    assert m.model == "Qwen/Qwen3.5-9B"
    assert m.ready is True
    assert m.status == "ready"


def test_model_info_unavailable():
    m = ModelInfo(model="some-model", ready=False, status="unavailable")
    assert m.ready is False
    assert m.status == "unavailable"
