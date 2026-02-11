"""Tests for overshoot.types — dataclass construction and helpers."""
import json

from overshoot import (
    CameraSource,
    ClipProcessingConfig,
    FeedbackCreateRequest,
    FeedbackResponse,
    FileSource,
    FrameProcessingConfig,
    InferenceConfig,
    KeepaliveResponse,
    Lease,
    LiveKitSource,
    ModelInfo,
    StatusResponse,
    StreamConfigResponse,
    StreamCreateResponse,
    StreamInferenceResult,
    TurnServer,
    WebRTCAnswer,
    WebRTCSource,
)


def test_livekit_source():
    s = LiveKitSource(url="wss://lk.example.com", token="tok-123")
    assert s.url == "wss://lk.example.com"
    assert s.token == "tok-123"


def test_webrtc_source():
    s = WebRTCSource(sdp="v=0\r\noffer")
    assert s.sdp == "v=0\r\noffer"


def test_file_source_defaults():
    s = FileSource(path="./video.mp4")
    assert s.path == "./video.mp4"
    assert s.loop is False


def test_file_source_loop():
    s = FileSource(path="./video.mp4", loop=True)
    assert s.loop is True


def test_camera_source_default():
    s = CameraSource()
    assert s.device == "default"


def test_camera_source_custom():
    s = CameraSource(device="/dev/video1")
    assert s.device == "/dev/video1"


def test_clip_processing_config():
    c = ClipProcessingConfig(sampling_ratio=0.5, fps=30)
    assert c.sampling_ratio == 0.5
    assert c.fps == 30
    assert c.clip_length_seconds == 1.0
    assert c.delay_seconds == 1.0


def test_frame_processing_config():
    f = FrameProcessingConfig(interval_seconds=3.0)
    assert f.interval_seconds == 3.0


def test_inference_config_defaults():
    i = InferenceConfig(prompt="Describe")
    assert i.prompt == "Describe"
    assert i.backend == "overshoot"
    assert i.model == "Qwen/Qwen3-VL-30B-A3B-Instruct"
    assert i.output_schema_json is None
    assert i.max_output_tokens is None


def test_inference_config_max_output_tokens():
    i = InferenceConfig(prompt="Describe", max_output_tokens=512)
    assert i.max_output_tokens == 512


def test_stream_create_response():
    r = StreamCreateResponse(
        stream_id="abc-123",
        webrtc=WebRTCAnswer(type="answer", sdp="v=0\r\nanswer"),
        lease=Lease(ttl_seconds=30),
        turn_servers=[TurnServer(urls="turn:example.com", username="u", credential="c")],
    )
    assert r.stream_id == "abc-123"
    assert r.webrtc is not None
    assert r.webrtc.sdp == "v=0\r\nanswer"
    assert r.lease is not None
    assert r.lease.ttl_seconds == 30
    assert r.turn_servers is not None
    assert len(r.turn_servers) == 1


def test_keepalive_response():
    k = KeepaliveResponse(status="ok", stream_id="abc", ttl_seconds=30, cost_cents=0.05)
    assert k.ttl_seconds == 30
    assert k.cost_cents == 0.05
    assert k.credits_remaining_cents is None
    assert k.seconds_charged is None


def test_keepalive_response_seconds_charged():
    k = KeepaliveResponse(
        status="ok", stream_id="abc", ttl_seconds=30, seconds_charged=5.2
    )
    assert k.seconds_charged == 5.2


def test_stream_config_response():
    c = StreamConfigResponse(
        id="cfg-1", stream_id="str-1", prompt="test", backend="overshoot", model="Qwen"
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
        model_backend="overshoot",
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
        model_backend="overshoot",
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
        model_backend="gemini",
        model_name="gemini-pro",
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
        model_backend="overshoot",
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
        model_backend="overshoot",
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
    """Dataclasses should be immutable."""
    s = LiveKitSource(url="wss://test", token="tok")
    try:
        s.url = "changed"  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass


def test_model_info():
    m = ModelInfo(model="Qwen/Qwen3-VL-30B-A3B-Instruct", ready=True, status="ready")
    assert m.model == "Qwen/Qwen3-VL-30B-A3B-Instruct"
    assert m.ready is True
    assert m.status == "ready"


def test_model_info_unavailable():
    m = ModelInfo(model="some-model", ready=False, status="unavailable")
    assert m.ready is False
    assert m.status == "unavailable"


def test_feedback_create_request():
    f = FeedbackCreateRequest(rating=5, category="quality", feedback="Great results")
    assert f.rating == 5
    assert f.category == "quality"
    assert f.feedback == "Great results"


def test_feedback_response():
    f = FeedbackResponse(
        id="fb-1",
        stream_id="str-1",
        rating=4,
        category="accuracy",
        feedback="Mostly correct",
        created_at="2025-02-10T00:00:00Z",
    )
    assert f.id == "fb-1"
    assert f.stream_id == "str-1"
    assert f.rating == 4
    assert f.created_at == "2025-02-10T00:00:00Z"


def test_feedback_response_defaults():
    f = FeedbackResponse(
        id="fb-2", stream_id="str-2", rating=3, category="speed", feedback="Slow"
    )
    assert f.created_at is None
