import logging
from dataclasses import asdict
from typing import Any, Optional

from ._constants import DEFAULT_BASE_URL
from ._http import HttpClient
from .types import (
    InferenceConfig,
    KeepaliveResponse,
    Lease,
    LiveKitConnection,
    LiveKitSource,
    ModelInfo,
    ProcessingConfig,
    StatusResponse,
    StreamConfigResponse,
    StreamCreateResponse,
    StreamMode,
    WireSource,
)

logger = logging.getLogger("overshoot")


def _serialize_source(source: Optional[WireSource]) -> Optional[dict[str, Any]]:
    """Convert a wire-ready source to the API payload format.

    Returns None when source is None (server creates LiveKit room).
    """
    if source is None:
        return None
    if isinstance(source, LiveKitSource):
        return {"type": "livekit", "url": source.url, "token": source.token}
    raise TypeError(f"Unsupported wire source type: {type(source)}")


def _serialize_processing(processing: ProcessingConfig) -> dict[str, Any]:
    """Convert a ProcessingConfig to the API payload format, stripping None values."""
    d = asdict(processing)
    return {k: v for k, v in d.items() if v is not None}


def _serialize_inference(inference: InferenceConfig) -> dict[str, Any]:
    """Convert an InferenceConfig to the API payload format, stripping None values."""
    d = asdict(inference)
    return {k: v for k, v in d.items() if v is not None}


def _parse_create_response(data: dict[str, Any]) -> StreamCreateResponse:
    """Parse the raw JSON dict into a StreamCreateResponse."""
    lease = None
    if data.get("lease"):
        lease = Lease(**data["lease"])

    livekit = None
    if data.get("livekit"):
        livekit = LiveKitConnection(**data["livekit"])

    return StreamCreateResponse(
        stream_id=data["stream_id"],
        lease=lease,
        livekit=livekit,
    )


def _parse_keepalive_response(data: dict[str, Any]) -> KeepaliveResponse:
    """Parse the raw JSON dict into a KeepaliveResponse."""
    return KeepaliveResponse(
        status=data.get("status", "ok"),
        stream_id=data.get("stream_id", ""),
        ttl_seconds=data.get("ttl_seconds", 0),
        credits_remaining_cents=data.get("credits_remaining_cents"),
        cost_cents=data.get("cost_cents"),
        seconds_charged=data.get("seconds_charged"),
        livekit_token=data.get("livekit_token"),
    )


def _parse_config_response(data: dict[str, Any]) -> StreamConfigResponse:
    """Parse the raw JSON dict into a StreamConfigResponse."""
    return StreamConfigResponse(
        id=data["id"],
        stream_id=data["stream_id"],
        prompt=data["prompt"],
        model=data["model"],
        output_schema_json=data.get("output_schema_json"),
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
    )


class ApiClient:
    """Low-level async HTTP client for the Overshoot Media Gateway API.

    Maps 1:1 to API endpoints. Returns typed response dataclasses.
    No background tasks, no WebSocket, no callbacks.

    For the high-level experience with automatic keepalive and result
    streaming, use :class:`overshoot.Overshoot` instead.

    Usage::

        api = overshoot.ApiClient(api_key="ovs_...")
        resp = await api.create_stream(source=..., processing=..., inference=...)
        await api.close_stream(resp.stream_id)
        await api.close()
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        self._http = HttpClient(api_key, base_url=base_url, timeout=timeout)

    async def create_stream(
        self,
        source: Optional[WireSource],
        processing: ProcessingConfig,
        inference: InferenceConfig,
        *,
        mode: Optional[StreamMode] = None,
    ) -> StreamCreateResponse:
        """POST /streams — Create a new analysis stream."""
        body: dict[str, Any] = {
            "processing": _serialize_processing(processing),
            "inference": _serialize_inference(inference),
        }
        serialized_source = _serialize_source(source)
        if serialized_source is not None:
            body["source"] = serialized_source
        if mode is not None:
            body["mode"] = mode

        data = await self._http.request("POST", "/streams", json_body=body)
        return _parse_create_response(data)

    async def keepalive(self, stream_id: str) -> KeepaliveResponse:
        """POST /streams/{stream_id}/keepalive — Renew stream lease."""
        data = await self._http.request("POST", f"/streams/{stream_id}/keepalive")
        return _parse_keepalive_response(data)

    async def update_prompt(self, stream_id: str, prompt: str) -> StreamConfigResponse:
        """PATCH /streams/{stream_id}/config/prompt — Update inference prompt."""
        data = await self._http.request(
            "PATCH",
            f"/streams/{stream_id}/config/prompt",
            json_body={"prompt": prompt},
        )
        return _parse_config_response(data)

    async def close_stream(self, stream_id: str) -> StatusResponse:
        """DELETE /streams/{stream_id} — Close stream and trigger final billing."""
        data = await self._http.request("DELETE", f"/streams/{stream_id}")
        return StatusResponse(status=data.get("status", "ok"))

    async def get_models(self) -> list[ModelInfo]:
        """GET /models — List available models and their status."""
        data = await self._http.request("GET", "/models")
        models: list[dict[str, Any]] = data if isinstance(data, list) else data.get("models", [])
        return [
            ModelInfo(
                model=m["model"],
                ready=m["ready"],
                status=m["status"],
            )
            for m in models
        ]

    async def health_check(self) -> str:
        """GET /healthz — Check API health."""
        data = await self._http.request("GET", "/healthz")
        status: str = data.get("status", "ok") if isinstance(data, dict) else "ok"
        return status

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        await self._http.close()
