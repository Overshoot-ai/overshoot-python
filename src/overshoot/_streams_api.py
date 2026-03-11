import logging
from typing import Any, Callable, Optional

from ._api_client import (
    _parse_create_response,
    _serialize_inference,
    _serialize_processing,
    _serialize_source,
)
from ._constants import (
    DEFAULT_CLIP_LENGTH_SECONDS,
    DEFAULT_DELAY_SECONDS,
    DEFAULT_INTERVAL_SECONDS,
    DEFAULT_TARGET_FPS,
)
from ._http import HttpClient
from ._livekit_transport import LiveKitTransport
from ._sources import ResolvedSource, resolve_source
from ._stream import Stream
from .types import (
    ClipProcessingConfig,
    FrameProcessingConfig,
    InferenceConfig,
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
        model: str,
        on_result: Callable[[StreamInferenceResult], Any],
        *,
        on_error: Optional[Callable[[Exception], Any]] = None,
        mode: Optional[StreamMode] = None,
        output_schema: Optional[dict[str, Any]] = None,
        max_output_tokens: Optional[int] = None,
        # Clip mode params
        target_fps: Optional[int] = None,
        clip_length_seconds: Optional[float] = None,
        delay_seconds: Optional[float] = None,
        # Frame mode params
        interval_seconds: Optional[float] = None,
    ) -> Stream:
        """Create and start a new analysis stream.

        Resolves the source (creating an FFmpeg pipeline and LiveKit video
        track for local sources), calls the API, starts background keepalive
        and WebSocket consumer tasks, and returns a running :class:`Stream`.

        Parameters
        ----------
        source:
            Video source — ``CameraSource``, ``FileSource``,
            ``RTSPSource``, ``HLSSource``, ``RTMPSource``,
            ``FrameSource``, or ``LiveKitSource``.
        prompt:
            The analysis task to run on each video segment.
        model:
            Model name for inference. Use ``get_models()`` to list
            available models.
        on_result:
            Callback invoked for each inference result.
        on_error:
            Optional callback for errors (keepalive failure, WS errors).
        mode:
            ``"clip"`` or ``"frame"``. Auto-detected from params if not set.
        output_schema:
            Optional JSON schema for structured output.
        max_output_tokens:
            Cap tokens per inference request.
        target_fps:
            Clip mode: target frame sampling rate (1-30).
        clip_length_seconds:
            Clip mode: duration of each clip.
        delay_seconds:
            Clip mode: delay between clips.
        interval_seconds:
            Frame mode: seconds between frame captures.
        """
        # 1. Build processing config (needed before resolve for target_fps)
        processing = self._build_processing(
            mode=mode,
            target_fps=target_fps,
            clip_length_seconds=clip_length_seconds,
            delay_seconds=delay_seconds,
            interval_seconds=interval_seconds,
        )

        # Determine effective target_fps for FFmpeg source
        effective_fps = DEFAULT_TARGET_FPS
        if isinstance(processing, ClipProcessingConfig) and processing.target_fps:
            effective_fps = processing.target_fps

        # 2. Resolve source (FFmpeg decode, LiveKit track creation, etc.)
        resolved = await resolve_source(source, target_fps=effective_fps)

        # 3. Determine mode
        if mode is None:
            mode = "frame" if isinstance(processing, FrameProcessingConfig) else "clip"

        # 4. Build inference config
        inference = InferenceConfig(
            prompt=prompt,
            model=model,
            output_schema_json=output_schema,
            max_output_tokens=max_output_tokens,
        )

        # 5. Call API
        body: dict[str, Any] = {
            "processing": _serialize_processing(processing),
            "inference": _serialize_inference(inference),
            "mode": mode,
        }
        serialized_source = _serialize_source(resolved.wire_source)
        if serialized_source is not None:
            body["source"] = serialized_source

        data = await self._http.request("POST", "/streams", json_body=body)
        response = _parse_create_response(data)

        logger.info("Stream created: %s", response.stream_id)

        # 6. Connect LiveKit transport if source uses native track publishing
        livekit_transport: Optional[LiveKitTransport] = None

        if response.livekit and resolved.is_native:
            livekit_transport = LiveKitTransport(
                on_fatal_error=on_error,
            )
            await livekit_transport.connect(
                url=response.livekit.url,
                token=response.livekit.token,
                video_track=resolved.livekit_video_track,
                target_fps=effective_fps,
            )

        # 7. Build and start the Stream
        ttl = response.lease.ttl_seconds if response.lease else 0

        stream = Stream(
            stream_id=response.stream_id,
            http=self._http,
            resolved_source=resolved,
            ttl_seconds=ttl,
            on_result=on_result,
            on_error=on_error,
            livekit_transport=livekit_transport,
        )
        stream._start()

        return stream

    @staticmethod
    def _build_processing(
        *,
        mode: Optional[StreamMode],
        target_fps: Optional[int],
        clip_length_seconds: Optional[float],
        delay_seconds: Optional[float],
        interval_seconds: Optional[float],
    ) -> ProcessingConfig:
        """Build the processing config from flat params."""
        # Explicit clip mode or clip-specific params provided
        has_clip_params = any(p is not None for p in (target_fps, clip_length_seconds, delay_seconds))
        if mode == "clip" or (mode is None and has_clip_params):
            return ClipProcessingConfig(
                target_fps=target_fps if target_fps is not None else DEFAULT_TARGET_FPS,
                clip_length_seconds=(
                    clip_length_seconds
                    if clip_length_seconds is not None
                    else DEFAULT_CLIP_LENGTH_SECONDS
                ),
                delay_seconds=(
                    delay_seconds if delay_seconds is not None else DEFAULT_DELAY_SECONDS
                ),
            )

        # Frame mode (default)
        return FrameProcessingConfig(
            interval_seconds=interval_seconds or DEFAULT_INTERVAL_SECONDS,
        )
