"""Overshoot Python SDK — real-time video analysis."""

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
    LiveKitSource,
    WebRTCSource,
    SourceConfig,
    WireSource,
    # Processing
    ClipProcessingConfig,
    FrameProcessingConfig,
    ProcessingConfig,
    InferenceConfig,
    StreamMode,
    ModelBackend,
    FinishReason,
    ModelStatus,
    StreamStopReason,
    # Responses
    StreamCreateResponse,
    StreamInferenceResult,
    KeepaliveResponse,
    StreamConfigResponse,
    StatusResponse,
    WebRTCAnswer,
    TurnServer,
    Lease,
    ModelInfo,
    FeedbackCreateRequest,
    FeedbackResponse,
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
    StreamClosedError,
    WebSocketError,
)


class Overshoot:
    """Overshoot API client.

    High-level entry point for real-time video analysis. Use
    ``client.streams.create()`` to start a stream with automatic
    keepalive and WebSocket result delivery.

    Usage::

        import overshoot

        client = overshoot.Overshoot(api_key="sk-...")

        stream = await client.streams.create(
            source=overshoot.CameraSource(),
            prompt="Describe what you see",
            on_result=lambda r: print(r.result),
        )

        await stream.close()
        await client.close()
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
    # Sources
    "CameraSource",
    "FileSource",
    "LiveKitSource",
    "WebRTCSource",
    "SourceConfig",
    "WireSource",
    # Processing
    "ClipProcessingConfig",
    "FrameProcessingConfig",
    "ProcessingConfig",
    "InferenceConfig",
    "StreamMode",
    "ModelBackend",
    # Type aliases
    "FinishReason",
    "ModelStatus",
    "StreamStopReason",
    # Responses
    "StreamCreateResponse",
    "StreamInferenceResult",
    "KeepaliveResponse",
    "StreamConfigResponse",
    "StatusResponse",
    "WebRTCAnswer",
    "TurnServer",
    "Lease",
    "ModelInfo",
    "FeedbackCreateRequest",
    "FeedbackResponse",
    # Errors
    "OvershootError",
    "ApiError",
    "AuthenticationError",
    "ValidationError",
    "NotFoundError",
    "InsufficientCreditsError",
    "ServerError",
    "NetworkError",
    "StreamClosedError",
    "WebSocketError",
]
