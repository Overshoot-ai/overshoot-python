"""Tests for overshoot.ApiClient — HTTP client with mocked responses."""
import pytest
from aioresponses import aioresponses

from overshoot import (
    ApiClient,
    ClipProcessingConfig,
    FrameProcessingConfig,
    InferenceConfig,
    LiveKitSource,
    WebRTCSource,
)
from overshoot.errors import (
    AuthenticationError,
    InsufficientCreditsError,
    NotFoundError,
    ServerError,
    ValidationError,
)


@pytest.fixture
def base_url() -> str:
    return "https://test.overshoot.ai/api/v0.2"


@pytest.fixture
def api_key() -> str:
    return "test-key-123"


@pytest.fixture
def mock_api():
    with aioresponses() as m:
        yield m


@pytest.mark.asyncio
async def test_create_stream_webrtc(mock_api, api_key, base_url):
    mock_api.post(
        f"{base_url}/streams",
        payload={
            "stream_id": "str-abc",
            "webrtc": {"type": "answer", "sdp": "v=0\r\nanswer"},
            "lease": {"ttl_seconds": 30},
            "turn_servers": [
                {"urls": "turn:example.com:3478", "username": "u", "credential": "c"}
            ],
        },
        status=201,
    )

    api = ApiClient(api_key, base_url=base_url)
    try:
        resp = await api.create_stream(
            source=WebRTCSource(sdp="v=0\r\noffer"),
            processing=ClipProcessingConfig(sampling_ratio=0.5, fps=30),
            inference=InferenceConfig(prompt="Test prompt"),
        )
        assert resp.stream_id == "str-abc"
        assert resp.webrtc is not None
        assert resp.webrtc.sdp == "v=0\r\nanswer"
        assert resp.lease is not None
        assert resp.lease.ttl_seconds == 30
        assert resp.turn_servers is not None
        assert len(resp.turn_servers) == 1
    finally:
        await api.close()


@pytest.mark.asyncio
async def test_create_stream_livekit(mock_api, api_key, base_url):
    mock_api.post(
        f"{base_url}/streams",
        payload={"stream_id": "str-lk", "lease": {"ttl_seconds": 60}},
        status=201,
    )

    api = ApiClient(api_key, base_url=base_url)
    try:
        resp = await api.create_stream(
            source=LiveKitSource(url="wss://lk.test", token="tok"),
            processing=FrameProcessingConfig(interval_seconds=2.0),
            inference=InferenceConfig(prompt="Test"),
            mode="frame",
        )
        assert resp.stream_id == "str-lk"
        assert resp.webrtc is None
    finally:
        await api.close()


@pytest.mark.asyncio
async def test_keepalive(mock_api, api_key, base_url):
    mock_api.post(
        f"{base_url}/streams/str-1/keepalive",
        payload={
            "status": "ok",
            "stream_id": "str-1",
            "ttl_seconds": 30,
            "credits_remaining_cents": 500,
            "cost_cents": 0.05,
        },
    )

    api = ApiClient(api_key, base_url=base_url)
    try:
        resp = await api.keepalive("str-1")
        assert resp.status == "ok"
        assert resp.ttl_seconds == 30
        assert resp.credits_remaining_cents == 500
        assert resp.cost_cents == 0.05
        assert resp.seconds_charged is None
    finally:
        await api.close()


@pytest.mark.asyncio
async def test_keepalive_with_seconds_charged(mock_api, api_key, base_url):
    mock_api.post(
        f"{base_url}/streams/str-1/keepalive",
        payload={
            "status": "ok",
            "stream_id": "str-1",
            "ttl_seconds": 30,
            "seconds_charged": 15.5,
        },
    )

    api = ApiClient(api_key, base_url=base_url)
    try:
        resp = await api.keepalive("str-1")
        assert resp.seconds_charged == 15.5
    finally:
        await api.close()


@pytest.mark.asyncio
async def test_update_prompt(mock_api, api_key, base_url):
    mock_api.patch(
        f"{base_url}/streams/str-1/config/prompt",
        payload={
            "id": "cfg-1",
            "stream_id": "str-1",
            "prompt": "new prompt",
            "backend": "overshoot",
            "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        },
    )

    api = ApiClient(api_key, base_url=base_url)
    try:
        resp = await api.update_prompt("str-1", "new prompt")
        assert resp.prompt == "new prompt"
        assert resp.stream_id == "str-1"
    finally:
        await api.close()


@pytest.mark.asyncio
async def test_close_stream(mock_api, api_key, base_url):
    mock_api.delete(
        f"{base_url}/streams/str-1",
        payload={"status": "ok"},
    )

    api = ApiClient(api_key, base_url=base_url)
    try:
        resp = await api.close_stream("str-1")
        assert resp.status == "ok"
    finally:
        await api.close()


# ── New endpoint tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_models(mock_api, api_key, base_url):
    mock_api.get(
        f"{base_url}/models",
        payload=[
            {"model": "Qwen/Qwen3-VL-30B-A3B-Instruct", "ready": True, "status": "ready"},
            {"model": "Qwen/Qwen3-VL-8B-Instruct", "ready": False, "status": "unavailable"},
        ],
    )

    api = ApiClient(api_key, base_url=base_url)
    try:
        models = await api.get_models()
        assert len(models) == 2
        assert models[0].model == "Qwen/Qwen3-VL-30B-A3B-Instruct"
        assert models[0].ready is True
        assert models[0].status == "ready"
        assert models[1].ready is False
        assert models[1].status == "unavailable"
    finally:
        await api.close()


@pytest.mark.asyncio
async def test_health_check(mock_api, api_key, base_url):
    mock_api.get(
        f"{base_url}/healthz",
        payload={"status": "ok"},
    )

    api = ApiClient(api_key, base_url=base_url)
    try:
        status = await api.health_check()
        assert status == "ok"
    finally:
        await api.close()


@pytest.mark.asyncio
async def test_submit_feedback(mock_api, api_key, base_url):
    mock_api.post(
        f"{base_url}/streams/str-1/feedback",
        payload={"status": "ok"},
    )

    api = ApiClient(api_key, base_url=base_url)
    try:
        resp = await api.submit_feedback(
            stream_id="str-1",
            rating=5,
            category="quality",
            feedback="Excellent results",
        )
        assert resp.status == "ok"
    finally:
        await api.close()


@pytest.mark.asyncio
async def test_get_feedback(mock_api, api_key, base_url):
    mock_api.get(
        f"{base_url}/streams/feedback",
        payload=[
            {
                "id": "fb-1",
                "stream_id": "str-1",
                "rating": 5,
                "category": "quality",
                "feedback": "Great",
                "created_at": "2025-02-10T00:00:00Z",
            },
        ],
    )

    api = ApiClient(api_key, base_url=base_url)
    try:
        feedback = await api.get_feedback()
        assert len(feedback) == 1
        assert feedback[0].id == "fb-1"
        assert feedback[0].stream_id == "str-1"
        assert feedback[0].rating == 5
        assert feedback[0].category == "quality"
        assert feedback[0].feedback == "Great"
        assert feedback[0].created_at == "2025-02-10T00:00:00Z"
    finally:
        await api.close()


# ── Error mapping tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_401_raises_authentication_error(mock_api, api_key, base_url):
    mock_api.post(
        f"{base_url}/streams",
        payload={"error": "unauthorized", "message": "Invalid API key"},
        status=401,
    )

    api = ApiClient(api_key, base_url=base_url)
    try:
        with pytest.raises(AuthenticationError):
            await api.create_stream(
                source=WebRTCSource(sdp="v=0"),
                processing=ClipProcessingConfig(sampling_ratio=0.5, fps=30),
                inference=InferenceConfig(prompt="test"),
            )
    finally:
        await api.close()


@pytest.mark.asyncio
async def test_402_raises_insufficient_credits(mock_api, api_key, base_url):
    mock_api.post(
        f"{base_url}/streams",
        payload={"error": "insufficient_credits", "message": "No credits"},
        status=402,
    )

    api = ApiClient(api_key, base_url=base_url)
    try:
        with pytest.raises(InsufficientCreditsError):
            await api.create_stream(
                source=WebRTCSource(sdp="v=0"),
                processing=ClipProcessingConfig(sampling_ratio=0.5, fps=30),
                inference=InferenceConfig(prompt="test"),
            )
    finally:
        await api.close()


@pytest.mark.asyncio
async def test_404_raises_not_found(mock_api, api_key, base_url):
    mock_api.post(
        f"{base_url}/streams/str-999/keepalive",
        payload={"error": "not_found", "message": "Stream not found"},
        status=404,
    )

    api = ApiClient(api_key, base_url=base_url)
    try:
        with pytest.raises(NotFoundError):
            await api.keepalive("str-999")
    finally:
        await api.close()


@pytest.mark.asyncio
async def test_422_raises_validation_error(mock_api, api_key, base_url):
    mock_api.post(
        f"{base_url}/streams",
        payload={"error": "validation_error", "message": "Invalid params"},
        status=422,
    )

    api = ApiClient(api_key, base_url=base_url)
    try:
        with pytest.raises(ValidationError):
            await api.create_stream(
                source=WebRTCSource(sdp="v=0"),
                processing=ClipProcessingConfig(sampling_ratio=0.5, fps=30),
                inference=InferenceConfig(prompt="test"),
            )
    finally:
        await api.close()


@pytest.mark.asyncio
async def test_500_raises_server_error(mock_api, api_key, base_url):
    mock_api.post(
        f"{base_url}/streams",
        payload={"error": "internal", "message": "Something broke"},
        status=500,
    )

    api = ApiClient(api_key, base_url=base_url)
    try:
        with pytest.raises(ServerError):
            await api.create_stream(
                source=WebRTCSource(sdp="v=0"),
                processing=ClipProcessingConfig(sampling_ratio=0.5, fps=30),
                inference=InferenceConfig(prompt="test"),
            )
    finally:
        await api.close()
