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
from ._go_publisher import GoPublisherSource, go_publisher_available, _probe_codec
from ._http import HttpClient
from ._livekit_transport import LiveKitTransport
from ._sources import ResolvedSource, resolve_source, _clamp_fps, _LOW_LATENCY_FLAGS
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

        # 2. Check if Go publisher is available and source is eligible.
        #    Only use Go for H.264 sources — H.265 transcode has pacing issues.
        use_go = self._should_use_go_publisher(source)
        if use_go:
            from ._go_publisher import _probe_codec
            from .types import RTSPSource as _RTSP
            probe_url = source.url if isinstance(source, _RTSP) else None
            if probe_url:
                codec = await _probe_codec(probe_url)
                if codec and codec != "h264":
                    logger.info("Source codec %s is not H.264 — using Python path", codec)
                    use_go = False

        # 3. Resolve source (skip for Go publisher — it handles FFmpeg+LiveKit internally)
        if use_go:
            resolved = ResolvedSource(wire_source=None)
        else:
            resolved = await resolve_source(source, target_fps=effective_fps)

        # 4. Determine mode
        if mode is None:
            mode = "frame" if isinstance(processing, FrameProcessingConfig) else "clip"

        # 5. Build inference config
        inference = InferenceConfig(
            prompt=prompt,
            model=model,
            output_schema_json=output_schema,
            max_output_tokens=max_output_tokens,
        )

        # 6. Call API
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

        # 7. Start Go publisher OR connect LiveKit transport
        livekit_transport: Optional[LiveKitTransport] = None

        if use_go and response.livekit:
            await self._start_go_publisher(
                source, resolved, response.livekit.url, response.livekit.token,
                effective_fps,
            )
        elif response.livekit and resolved.is_native:
            livekit_transport = LiveKitTransport(
                on_fatal_error=on_error,
            )
            await livekit_transport.connect(
                url=response.livekit.url,
                token=response.livekit.token,
                video_track=resolved.livekit_video_track,
                target_fps=effective_fps,
            )

        # 8. Build and start the Stream
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
    def _should_use_go_publisher(source: SourceConfig) -> bool:
        """Check if the Go publisher should be used for this source."""
        from .types import FileSource, RTSPSource, HLSSource, RTMPSource
        if not go_publisher_available():
            return False
        # Go publisher only handles FFmpeg-based network/file sources
        return isinstance(source, (FileSource, RTSPSource, HLSSource, RTMPSource))

    @staticmethod
    async def _start_go_publisher(
        source: SourceConfig,
        resolved: ResolvedSource,
        livekit_url: str,
        livekit_token: str,
        target_fps: int,
    ) -> None:
        """Start the Go publisher pipeline and attach it to the resolved source."""
        from .types import FileSource, RTSPSource, HLSSource, RTMPSource

        # Determine input path and FFmpeg flags
        if isinstance(source, FileSource):
            input_path = source.path
            extra_input_args = None
            loop = source.loop
        elif isinstance(source, RTSPSource):
            input_path = source.url
            extra_input_args = [
                *_LOW_LATENCY_FLAGS,
                "-rtsp_transport", "tcp", "-rtsp_flags", "prefer_tcp",
            ]
            loop = False
        elif isinstance(source, HLSSource):
            input_path = source.url
            extra_input_args = list(_LOW_LATENCY_FLAGS)
            loop = False
        elif isinstance(source, RTMPSource):
            input_path = source.url
            extra_input_args = list(_LOW_LATENCY_FLAGS)
            loop = False
        else:
            raise TypeError(f"Go publisher does not support {type(source)}")

        # Detect source codec for passthrough optimization
        source_codec = await _probe_codec(input_path)
        if source_codec:
            logger.info("Detected source codec: %s", source_codec)

        ffmpeg_fps = _clamp_fps(target_fps)

        go_pub = GoPublisherSource(
            input_path,
            livekit_url=livekit_url,
            livekit_token=livekit_token,
            target_fps=ffmpeg_fps,
            source_codec=source_codec,
            extra_input_args=extra_input_args,
            loop=loop,
        )
        await go_pub.start()
        resolved.go_publisher = go_pub

        logger.info("Go publisher pipeline started (codec=%s, fps=%d)",
                     source_codec or "transcode", ffmpeg_fps)

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
