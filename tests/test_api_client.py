"""Tests for overshoot.ApiClient — HTTP client with mocked responses."""
import pytest
from aioresponses import aioresponses

from overshoot import (
    ApiClient,
    ClipProcessingConfig,
    FrameProcessingConfig,
    InferenceConfig,
    LiveKitSource,
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
async def test_create_stream_livekit(mock_api, api_key, base_url):
    mock_api.post(
        f"{base_url}/streams",
        payload={
            "stream_id": "str-lk",
            "livekit": {"url": "wss://lk.test", "token": "tok"},
            "lease": {"ttl_seconds": 45},
        },
        status=201,
    )

    api = ApiClient(api_key, base_url=base_url)
    try:
        resp = await api.create_stream(
            source=LiveKitSource(url="wss://lk.test", token="tok"),
            processing=FrameProcessingConfig(interval_seconds=2.0),
            inference=InferenceConfig(prompt="Test", model="Qwen/Qwen3.5-9B"),
            mode="frame",
        )
        assert resp.stream_id == "str-lk"
        assert resp.livekit is not None
        assert resp.livekit.url == "wss://lk.test"
    finally:
        await api.close()


@pytest.mark.asyncio
async def test_create_stream_native(mock_api, api_key, base_url):
    mock_api.post(
        f"{base_url}/streams",
        payload={
            "stream_id": "str-native",
            "livekit": {"url": "wss://lk.overshoot.ai", "token": "server-tok"},
            "lease": {"ttl_seconds": 45},
        },
        status=201,
    )

    api = ApiClient(api_key, base_url=base_url)
    try:
        resp = await api.create_stream(
            source=None,
            processing=ClipProcessingConfig(target_fps=6),
            inference=InferenceConfig(prompt="Test", model="Qwen/Qwen3.5-9B"),
            mode="clip",
        )
        assert resp.stream_id == "str-native"
        assert resp.livekit is not None
    finally:
        await api.close()


@pytest.mark.asyncio
async def test_keepalive(mock_api, api_key, base_url):
    mock_api.post(
        f"{base_url}/streams/str-1/keepalive",
        payload={
            "status": "ok",
            "stream_id": "str-1",
            "ttl_seconds": 45,
            "credits_remaining_cents": 500,
            "cost_cents": 0.05,
        },
    )

    api = ApiClient(api_key, base_url=base_url)
    try:
        resp = await api.keepalive("str-1")
        assert resp.status == "ok"
        assert resp.ttl_seconds == 45
        assert resp.credits_remaining_cents == 500
        assert resp.cost_cents == 0.05
        assert resp.seconds_charged is None
    finally:
        await api.close()


@pytest.mark.asyncio
async def test_keepalive_with_livekit_token(mock_api, api_key, base_url):
    mock_api.post(
        f"{base_url}/streams/str-1/keepalive",
        payload={
            "status": "ok",
            "stream_id": "str-1",
            "ttl_seconds": 45,
            "livekit_token": "refreshed-token",
        },
    )

    api = ApiClient(api_key, base_url=base_url)
    try:
        resp = await api.keepalive("str-1")
        assert resp.livekit_token == "refreshed-token"
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
            "model": "Qwen/Qwen3.5-9B",
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


@pytest.mark.asyncio
async def test_get_models(mock_api, api_key, base_url):
    mock_api.get(
        f"{base_url}/models",
        payload=[
            {"model": "Qwen/Qwen3.5-9B", "ready": True, "status": "ready"},
            {"model": "Qwen/Qwen3-VL-8B-Instruct", "ready": False, "status": "unavailable"},
        ],
    )

    api = ApiClient(api_key, base_url=base_url)
    try:
        models = await api.get_models()
        assert len(models) == 2
        assert models[0].model == "Qwen/Qwen3.5-9B"
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
                source=LiveKitSource(url="wss://test", token="tok"),
                processing=ClipProcessingConfig(target_fps=6),
                inference=InferenceConfig(prompt="test", model="test-model"),
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
                source=LiveKitSource(url="wss://test", token="tok"),
                processing=ClipProcessingConfig(target_fps=6),
                inference=InferenceConfig(prompt="test", model="test-model"),
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
                source=LiveKitSource(url="wss://test", token="tok"),
                processing=ClipProcessingConfig(target_fps=6),
                inference=InferenceConfig(prompt="test", model="test-model"),
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
                source=LiveKitSource(url="wss://test", token="tok"),
                processing=ClipProcessingConfig(target_fps=6),
                inference=InferenceConfig(prompt="test", model="test-model"),
            )
    finally:
        await api.close()
