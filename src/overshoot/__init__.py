"""Overshoot Python SDK — real-time video analysis."""

from typing import Any, Optional

from ._version import __version__
from ._constants import DEFAULT_BASE_URL
from ._http import HttpClient
from ._api_client import ApiClient
from ._stream import Stream
from ._streams_api import StreamsAPI

# Source types
from .types import (
    CameraSource,
    FileSource,
    FrameSource,
    HLSSource,
    LiveKitSource,
    RTMPSource,
    RTSPSource,
    SourceConfig,
    WireSource,
    # Processing
    ClipProcessingConfig,
    FrameProcessingConfig,
    ProcessingConfig,
    InferenceConfig,
    StreamMode,
    FinishReason,
    ModelStatus,
    StreamStopReason,
    # Responses
    LiveKitConnection,
    StreamCreateResponse,
    StreamInferenceResult,
    KeepaliveResponse,
    StreamConfigResponse,
    StatusResponse,
    Lease,
    ModelInfo,
    ReinferResult,
)

# Errors
from .errors import (
    OvershootError,
    ApiError,
    AuthenticationError,
    ValidationError,
    NotFoundError,
    InsufficientCreditsError,
    ServerError,
    NetworkError,
    SourceEndedError,
    StreamClosedError,
    WebSocketError,
)


from ._go_publisher import go_publisher_available


async def get_models(
    api_key: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 30.0,
) -> list[ModelInfo]:
    """Fetch available models and their status.

    Convenience function that creates a temporary client, fetches
    models, and closes the client. For repeated calls, use
    :meth:`ApiClient.get_models` directly.

    Usage::

        models = await overshoot.get_models(api_key="ovs_...")
        ready = [m for m in models if m.ready]
    """
    client = ApiClient(api_key, base_url=base_url, timeout=timeout)
    try:
        return await client.get_models()
    finally:
        await client.close()


class Overshoot:
    """Overshoot API client — high-level entry point.

    Manages an HTTP session and provides ``client.streams`` for creating
    streams with automatic keepalive and WebSocket result delivery.

    Usage::

        import overshoot

        client = overshoot.Overshoot(api_key="ovs_...")

        stream = await client.streams.create(
            source=overshoot.CameraSource(),
            prompt="Describe what you see",
            model="Qwen/Qwen3.5-9B",
            on_result=lambda r: print(r.result),
        )

        await stream.close()
        await client.close()

    For direct HTTP control without background tasks, use
    :class:`ApiClient` instead.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        self._http = HttpClient(api_key, base_url=base_url, timeout=timeout)
        self.streams = StreamsAPI(self._http)

    async def reinfer(
        self,
        result_id: str,
        prompt: str,
        model: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        output_schema: Optional[dict[str, Any]] = None,
    ) -> ReinferResult:
        """Re-run inference on a persisted clip with a new prompt.

        Parameters
        ----------
        result_id:
            The ``id`` from a :class:`StreamInferenceResult`.
        prompt:
            New prompt to run on the clip.
        model:
            Model name for inference.
        temperature:
            Optional sampling temperature.
        max_tokens:
            Optional max output tokens.
        output_schema:
            Optional JSON schema for structured output.

        Returns
        -------
        ReinferResult
            The re-inference result.
        """
        api = ApiClient.__new__(ApiClient)
        api._http = self._http
        return await api.reinfer(
            result_id, prompt, model,
            temperature=temperature,
            max_tokens=max_tokens,
            output_schema=output_schema,
        )

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        await self._http.close()


__all__ = [
    # Version
    "__version__",
    # Clients
    "Overshoot",
    "ApiClient",
    "Stream",
    # Utility
    "get_models",
    "go_publisher_available",
    # Sources
    "CameraSource",
    "FileSource",
    "FrameSource",
    "HLSSource",
    "LiveKitSource",
    "RTMPSource",
    "RTSPSource",
    "SourceConfig",
    "WireSource",
    # Processing
    "ClipProcessingConfig",
    "FrameProcessingConfig",
    "ProcessingConfig",
    "InferenceConfig",
    "StreamMode",
    # Type aliases
    "FinishReason",
    "ModelStatus",
    "StreamStopReason",
    # Responses
    "LiveKitConnection",
    "StreamCreateResponse",
    "StreamInferenceResult",
    "KeepaliveResponse",
    "StreamConfigResponse",
    "StatusResponse",
    "Lease",
    "ModelInfo",
    "ReinferResult",
    # Errors
    "OvershootError",
    "ApiError",
    "AuthenticationError",
    "ValidationError",
    "NotFoundError",
    "InsufficientCreditsError",
    "ServerError",
    "NetworkError",
    "SourceEndedError",
    "StreamClosedError",
    "WebSocketError",
]
