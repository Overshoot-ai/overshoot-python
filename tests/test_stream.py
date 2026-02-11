"""Tests for overshoot.Stream — lifecycle and background tasks."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from overshoot._http import HttpClient
from overshoot._sources import ResolvedSource
from overshoot._stream import Stream
from overshoot.errors import StreamClosedError
from overshoot.types import LiveKitSource


def _make_stream(
    api_key: str = "test-key",
    base_url: str = "https://test.overshoot.ai/api/v0.2",
    stream_id: str = "str-test",
    ttl_seconds: int = 30,
    on_result: MagicMock | None = None,
    on_error: MagicMock | None = None,
) -> tuple[Stream, HttpClient]:
    http = HttpClient(api_key, base_url=base_url)
    resolved = ResolvedSource(wire_source=LiveKitSource(url="wss://test", token="tok"))

    stream = Stream(
        stream_id=stream_id,
        http=http,
        resolved_source=resolved,
        ttl_seconds=ttl_seconds,
        on_result=on_result or MagicMock(),
        on_error=on_error or MagicMock(),
    )
    return stream, http


def test_stream_properties():
    stream, http = _make_stream(stream_id="str-123")
    assert stream.stream_id == "str-123"
    assert stream.is_active is True


@pytest.mark.asyncio
async def test_stream_close_is_idempotent():
    stream, http = _make_stream()

    # Mock the HTTP request for DELETE
    http.request = AsyncMock(return_value={"status": "ok"})

    await stream.close()
    assert stream.is_active is False

    # Second close should be a no-op
    await stream.close()
    # DELETE should only be called once
    http.request.assert_called_once()
    await http.close()


@pytest.mark.asyncio
async def test_update_prompt_when_closed_raises():
    stream, http = _make_stream()
    http.request = AsyncMock(return_value={"status": "ok"})
    await stream.close()

    with pytest.raises(StreamClosedError):
        await stream.update_prompt("new prompt")

    await http.close()


@pytest.mark.asyncio
async def test_update_prompt_calls_api():
    stream, http = _make_stream()
    http.request = AsyncMock(return_value={
        "id": "cfg-1",
        "stream_id": "str-test",
        "prompt": "updated",
        "backend": "overshoot",
        "model": "Qwen",
    })

    result = await stream.update_prompt("updated")
    assert result.prompt == "updated"

    http.request.assert_called_once_with(
        "PATCH",
        "/streams/str-test/config/prompt",
        json_body={"prompt": "updated"},
    )

    # Clean up without hitting the real API
    stream._closed = True
    await http.close()


def test_handle_ws_message_parses_finish_reason():
    on_result = MagicMock()
    stream, http = _make_stream(on_result=on_result)

    import json

    msg = json.dumps({
        "id": "res-1",
        "stream_id": "str-test",
        "mode": "clip",
        "model_backend": "overshoot",
        "model_name": "Qwen",
        "prompt": "test",
        "result": "hello",
        "inference_latency_ms": 100.0,
        "total_latency_ms": 150.0,
        "ok": True,
        "finish_reason": "stop",
    })
    stream._handle_ws_message(msg)

    on_result.assert_called_once()
    result = on_result.call_args[0][0]
    assert result.finish_reason == "stop"


def test_handle_ws_message_finish_reason_absent():
    on_result = MagicMock()
    stream, http = _make_stream(on_result=on_result)

    import json

    msg = json.dumps({
        "id": "res-2",
        "stream_id": "str-test",
        "mode": "frame",
        "model_backend": "overshoot",
        "model_name": "Qwen",
        "prompt": "test",
        "result": "world",
        "inference_latency_ms": 100.0,
        "total_latency_ms": 150.0,
        "ok": True,
    })
    stream._handle_ws_message(msg)

    on_result.assert_called_once()
    result = on_result.call_args[0][0]
    assert result.finish_reason is None
