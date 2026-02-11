import json
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union


# ── Stream mode / backend ────────────────────────────────────────────

StreamMode = Literal["clip", "frame"]
ModelBackend = Literal["overshoot", "gemini"]
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
    """LiveKit room as the video source. No extra dependencies required."""

    url: str
    token: str


@dataclass(frozen=True, slots=True)
class WebRTCSource:
    """Raw WebRTC SDP offer. User manages their own peer connection."""

    sdp: str


@dataclass(frozen=True, slots=True)
class FileSource:
    """Stream a local video file over WebRTC."""

    path: str
    loop: bool = False


@dataclass(frozen=True, slots=True)
class CameraSource:
    """Capture from a local camera over WebRTC."""

    device: str = "default"


SourceConfig = Union[LiveKitSource, WebRTCSource, FileSource, CameraSource]

# Wire-ready sources (accepted by ApiClient / sent directly to the API)
WireSource = Union[LiveKitSource, WebRTCSource]


# ── Processing configs ───────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ClipProcessingConfig:
    """Processing parameters for clip mode (temporal video analysis)."""

    sampling_ratio: float
    fps: int
    clip_length_seconds: float = 1.0
    delay_seconds: float = 1.0


@dataclass(frozen=True, slots=True)
class FrameProcessingConfig:
    """Processing parameters for frame mode (periodic snapshot analysis)."""

    interval_seconds: float = 2.0


ProcessingConfig = Union[ClipProcessingConfig, FrameProcessingConfig]


# ── Inference config ─────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Model inference configuration."""

    prompt: str
    backend: ModelBackend = "overshoot"
    model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    output_schema_json: Optional[dict[str, Any]] = None
    max_output_tokens: Optional[int] = None


# ── Response types ───────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class WebRTCAnswer:
    """WebRTC SDP answer returned from the server."""

    type: Literal["answer"]
    sdp: str


@dataclass(frozen=True, slots=True)
class TurnServer:
    """TURN/TURNS server configuration."""

    urls: str
    username: str
    credential: str


@dataclass(frozen=True, slots=True)
class Lease:
    """Stream lease information."""

    ttl_seconds: int


@dataclass(frozen=True, slots=True)
class StreamCreateResponse:
    """Response from POST /streams."""

    stream_id: str
    webrtc: Optional[WebRTCAnswer] = None
    lease: Optional[Lease] = None
    turn_servers: Optional[list[TurnServer]] = None


@dataclass(frozen=True, slots=True)
class KeepaliveResponse:
    """Response from POST /streams/{id}/keepalive."""

    status: str
    stream_id: str
    ttl_seconds: int
    credits_remaining_cents: Optional[float] = None
    cost_cents: Optional[float] = None
    seconds_charged: Optional[float] = None


@dataclass(frozen=True, slots=True)
class StreamConfigResponse:
    """Response from PATCH /streams/{id}/config/prompt."""

    id: str
    stream_id: str
    prompt: str
    backend: ModelBackend
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
    model_backend: ModelBackend
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


# ── Model info ──────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Information about an available model."""

    model: str
    ready: bool
    status: ModelStatus


# ── Feedback ────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class FeedbackCreateRequest:
    """Request body for submitting feedback on a stream."""

    rating: int
    category: str
    feedback: str


@dataclass(frozen=True, slots=True)
class FeedbackResponse:
    """Response from the feedback endpoints."""

    id: str
    stream_id: str
    rating: int
    category: str
    feedback: str
    created_at: Optional[str] = None
