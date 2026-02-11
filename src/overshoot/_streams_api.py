import logging
from typing import Any, Callable, Optional

from ._api_client import (
    _parse_create_response,
    _serialize_inference,
    _serialize_processing,
    _serialize_source,
)
from ._constants import (
    DEFAULT_BACKEND,
    DEFAULT_CLIP_LENGTH_SECONDS,
    DEFAULT_DELAY_SECONDS,
    DEFAULT_FPS,
    DEFAULT_INTERVAL_SECONDS,
    DEFAULT_MODEL,
    DEFAULT_SAMPLING_RATIO,
)
from ._http import HttpClient
from ._sources import resolve_source
from ._stream import Stream
from .types import (
    ClipProcessingConfig,
    FrameProcessingConfig,
    InferenceConfig,
    ModelBackend,
    ProcessingConfig,
    SourceConfig,
    StreamInferenceResult,
    StreamMode,
)

logger = logging.getLogger("overshoot")


class StreamsAPI:
    """Namespace for stream operations on the high-level Overshoot client.

    Accessed via ``client.streams``.
    """

    def __init__(self, http: HttpClient) -> None:
        self._http = http

    async def create(
        self,
        source: SourceConfig,
        prompt: str,
        on_result: Callable[[StreamInferenceResult], Any],
        *,
        on_error: Optional[Callable[[Exception], Any]] = None,
        mode: Optional[StreamMode] = None,
        backend: ModelBackend = DEFAULT_BACKEND,
        model: str = DEFAULT_MODEL,
        output_schema: Optional[dict[str, Any]] = None,
        max_output_tokens: Optional[int] = None,
        # Clip mode params
        sampling_ratio: Optional[float] = None,
        fps: Optional[int] = None,
        clip_length_seconds: Optional[float] = None,
        delay_seconds: Optional[float] = None,
        # Frame mode params
        interval_seconds: Optional[float] = None,
    ) -> Stream:
        """Create and start a new analysis stream.

        Resolves the source (creating a WebRTC peer connection if needed),
        calls the API, starts background keepalive and WebSocket consumer
        tasks, and returns a running :class:`Stream`.

        Parameters
        ----------
        source:
            Video source — ``CameraSource``, ``FileSource``,
            ``LiveKitSource``, or ``WebRTCSource``.
        prompt:
            The analysis task to run on each video segment.
        on_result:
            Callback invoked for each inference result.
        on_error:
            Optional callback for errors (keepalive failure, WS errors).
        mode:
            ``"clip"`` or ``"frame"``. Auto-detected from params if not set.
        backend:
            Model backend (``"overshoot"`` or ``"gemini"``).
        model:
            Model name for inference.
        output_schema:
            Optional JSON schema for structured output.
        sampling_ratio:
            Clip mode: fraction of frames to sample (0.0–1.0).
        fps:
            Clip mode: frames per second.
        clip_length_seconds:
            Clip mode: duration of each clip.
        delay_seconds:
            Clip mode: delay between clips.
        interval_seconds:
            Frame mode: seconds between frame captures.
        """
        # 1. Resolve source (may create WebRTC peer connection)
        resolved = await resolve_source(source)

        # 2. Build processing config
        processing = self._build_processing(
            mode=mode,
            sampling_ratio=sampling_ratio,
            fps=fps,
            clip_length_seconds=clip_length_seconds,
            delay_seconds=delay_seconds,
            interval_seconds=interval_seconds,
        )

        # 3. Determine mode
        if mode is None:
            mode = "frame" if isinstance(processing, FrameProcessingConfig) else "clip"

        # 4. Build inference config
        inference = InferenceConfig(
            prompt=prompt,
            backend=backend,
            model=model,
            output_schema_json=output_schema,
            max_output_tokens=max_output_tokens,
        )

        # 5. Call API
        body: dict[str, Any] = {
            "source": _serialize_source(resolved.wire_source),
            "processing": _serialize_processing(processing),
            "inference": _serialize_inference(inference),
            "mode": mode,
        }
        data = await self._http.request("POST", "/streams", json_body=body)
        response = _parse_create_response(data)

        logger.info("Stream created: %s", response.stream_id)

        # 6. Apply SDP answer if we have a peer connection
        if response.webrtc and resolved.peer_connection is not None:
            await resolved.apply_answer(response.webrtc)

        # 7. Build and start the Stream
        ttl = response.lease.ttl_seconds if response.lease else 0

        stream = Stream(
            stream_id=response.stream_id,
            http=self._http,
            resolved_source=resolved,
            ttl_seconds=ttl,
            on_result=on_result,
            on_error=on_error,
        )
        stream._start()

        return stream

    @staticmethod
    def _build_processing(
        *,
        mode: Optional[StreamMode],
        sampling_ratio: Optional[float],
        fps: Optional[int],
        clip_length_seconds: Optional[float],
        delay_seconds: Optional[float],
        interval_seconds: Optional[float],
    ) -> ProcessingConfig:
        """Build the processing config from flat params."""
        # Explicit frame mode or interval_seconds provided
        if mode == "frame" or (mode is None and interval_seconds is not None):
            return FrameProcessingConfig(
                interval_seconds=interval_seconds or DEFAULT_INTERVAL_SECONDS,
            )

        # Clip mode (default)
        return ClipProcessingConfig(
            sampling_ratio=sampling_ratio if sampling_ratio is not None else DEFAULT_SAMPLING_RATIO,
            fps=fps if fps is not None else DEFAULT_FPS,
            clip_length_seconds=(
                clip_length_seconds
                if clip_length_seconds is not None
                else DEFAULT_CLIP_LENGTH_SECONDS
            ),
            delay_seconds=(
                delay_seconds if delay_seconds is not None else DEFAULT_DELAY_SECONDS
            ),
        )
