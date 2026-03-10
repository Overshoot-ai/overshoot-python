import json
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union


# ── Stream mode ─────────────────────────────────────────────────────

StreamMode = Literal["clip", "frame"]
FinishReason = Literal["stop", "length", "content_filter"]
ModelStatus = Literal["unavailable", "ready", "degraded", "saturated"]
StreamStopReason = Literal[
    "client_requested",
    "webrtc_disconnected",
    "livekit_disconnected",
    "lease_expired",
    "insufficient_credits",
]


# ── Source types ─────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class LiveKitSource:
    """User-managed LiveKit room as the video source.

    Pass the LiveKit server URL and a publish-capable token.
    You are responsible for publishing video to the room.
    """

    url: str
    token: str


@dataclass(frozen=True, slots=True)
class FileSource:
    """Stream a local video file via FFmpeg.

    Set ``loop=True`` to loop continuously until the stream is closed.
    Requires ``ffmpeg`` on PATH.
    """

    path: str
    loop: bool = False


@dataclass(frozen=True, slots=True)
class RTSPSource:
    """Stream from an RTSP camera/server via FFmpeg.

    Uses TCP transport for reliability by default.
    Requires ``ffmpeg`` on PATH.
    """

    url: str


@dataclass(frozen=True, slots=True)
class HLSSource:
    """Stream from an HLS endpoint via FFmpeg.

    Requires ``ffmpeg`` on PATH.
    """

    url: str


@dataclass(frozen=True, slots=True)
class RTMPSource:
    """Stream from an RTMP endpoint via FFmpeg.

    Requires ``ffmpeg`` on PATH.
    """

    url: str


@dataclass(frozen=True, slots=True)
class CameraSource:
    """Capture from a local camera via FFmpeg.

    Requires ``ffmpeg`` on PATH.  Platform-specific device selection
    is handled automatically when ``device="default"``.

    Parameters
    ----------
    device:
        Camera device identifier. ``"default"`` auto-detects the
        platform camera (``/dev/video0`` on Linux, ``default`` via
        avfoundation on macOS, ``video=0`` via dshow on Windows).
    width:
        Capture width. Defaults to 1280.
    height:
        Capture height. Defaults to 720.
    """

    device: str = "default"
    width: int = 1280
    height: int = 720


@dataclass(slots=True)
class FrameSource:
    """Programmatic frame source — push raw video frames from your own pipeline.

    Create a ``FrameSource``, pass it to ``streams.create()``, then call
    :meth:`push_frame` to send frames. Works with raw RGBA bytes or
    numpy arrays (uint8, shape ``(height, width, 4)``).

    Usage::

        source = overshoot.FrameSource(width=640, height=480)
        stream = await client.streams.create(
            source=source,
            prompt="Describe what you see",
            model="Qwen/Qwen3.5-9B",
            on_result=lambda r: print(r.result),
        )

        # From OpenCV, PIL, or any pipeline
        source.push_frame(rgba_bytes)

    The ``FrameSource`` is mutable (not frozen) because the SDK
    attaches internal LiveKit objects to it after stream creation.
    """

    width: int
    height: int

    # Attached by the SDK during source resolution — not user-facing
    _livekit_video_source: Any = None
    _livekit_video_track: Any = None

    def push_frame(self, data: Any) -> None:
        """Push a single video frame.

        Parameters
        ----------
        data:
            Frame data — either raw RGBA bytes (length must be
            ``width * height * 4``) or a numpy ``ndarray`` with
            shape ``(height, width, 4)`` and dtype ``uint8``.

        Raises
        ------
        RuntimeError:
            If the source is not yet connected (stream not created).
        ValueError:
            If the data size does not match the expected frame size.
        """
        if self._livekit_video_source is None:
            raise RuntimeError(
                "FrameSource is not connected — create a stream first"
            )

        from livekit import rtc as livekit_rtc

        # Convert numpy array to bytes
        raw: bytes
        if hasattr(data, "tobytes"):
            raw = data.tobytes()
        elif isinstance(data, (bytes, bytearray, memoryview)):
            raw = bytes(data)
        else:
            raise TypeError(
                f"Expected bytes or numpy array, got {type(data).__name__}"
            )

        expected = self.width * self.height * 4
        if len(raw) != expected:
            raise ValueError(
                f"Frame data size mismatch: got {len(raw)} bytes, "
                f"expected {expected} ({self.width}x{self.height} RGBA)"
            )

        frame = livekit_rtc.VideoFrame(
            self.width,
            self.height,
            livekit_rtc.VideoBufferType.RGBA,
            raw,
        )
        self._livekit_video_source.capture_frame(frame)


# All source configs the high-level API accepts
SourceConfig = Union[
    LiveKitSource, FileSource, RTSPSource, HLSSource, RTMPSource,
    CameraSource, FrameSource,
]

# Wire-ready sources (sent directly to the API)
WireSource = LiveKitSource


# ── Processing configs ───────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ClipProcessingConfig:
    """Processing parameters for clip mode (temporal video analysis).

    ``target_fps`` controls the frame sampling rate on the server (1-30).
    """

    target_fps: Optional[int] = None
    clip_length_seconds: float = 0.5
    delay_seconds: float = 0.5


@dataclass(frozen=True, slots=True)
class FrameProcessingConfig:
    """Processing parameters for frame mode (periodic snapshot analysis)."""

    interval_seconds: float = 0.2


ProcessingConfig = Union[ClipProcessingConfig, FrameProcessingConfig]


# ── Inference config ─────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Model inference configuration."""

    prompt: str
    model: str
    output_schema_json: Optional[dict[str, Any]] = None
    max_output_tokens: Optional[int] = None


# ── Response types ───────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Lease:
    """Stream lease information."""

    ttl_seconds: int


@dataclass(frozen=True, slots=True)
class LiveKitConnection:
    """LiveKit room connection details returned for native transport streams."""

    url: str
    token: str


@dataclass(frozen=True, slots=True)
class StreamCreateResponse:
    """Response from POST /streams."""

    stream_id: str
    lease: Optional[Lease] = None
    livekit: Optional[LiveKitConnection] = None


@dataclass(frozen=True, slots=True)
class KeepaliveResponse:
    """Response from POST /streams/{id}/keepalive."""

    status: str
    stream_id: str
    ttl_seconds: int
    credits_remaining_cents: Optional[float] = None
    cost_cents: Optional[float] = None
    seconds_charged: Optional[float] = None
    livekit_token: Optional[str] = None


@dataclass(frozen=True, slots=True)
class StreamConfigResponse:
    """Response from PATCH /streams/{id}/config/prompt."""

    id: str
    stream_id: str
    prompt: str
    model: str
    output_schema_json: Optional[dict[str, Any]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass(frozen=True, slots=True)
class StatusResponse:
    """Response from DELETE /streams/{id}."""

    status: str


@dataclass(frozen=True, slots=True)
class StreamInferenceResult:
    """A single inference result pushed over the WebSocket."""

    id: str
    stream_id: str
    mode: StreamMode
    model_name: str
    prompt: str
    result: str
    inference_latency_ms: float
    total_latency_ms: float
    ok: bool
    error: Optional[str] = None
    finish_reason: Optional[str] = None

    def result_json(self) -> Any:
        """Parse ``self.result`` as JSON.

        Use this when the stream was created with ``output_schema`` —
        the model's output will be valid JSON.

        Raises ``ValueError`` if the result is not valid JSON.
        """
        return json.loads(self.result)


# ── Re-inference result ─────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ReinferResult:
    """Result from re-running inference on a persisted clip.

    Returned by :meth:`ApiClient.reinfer` and :meth:`Overshoot.reinfer`.
    """

    id: str
    model: str
    content: str
    finish_reason: Optional[str] = None
    usage: Optional[dict[str, Any]] = None

    def content_json(self) -> Any:
        """Parse ``self.content`` as JSON.

        Raises ``ValueError`` if the content is not valid JSON.
        """
        return json.loads(self.content)


# ── Model info ──────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Information about an available model."""

    model: str
    ready: bool
    status: ModelStatus
