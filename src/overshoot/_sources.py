"""Source resolution — converts source configs into LiveKit-publishable media.

FFmpeg-based sources (File, RTSP, HLS, RTMP, Camera) are decoded with an
FFmpeg subprocess and published to a server-managed LiveKit room.
FrameSource lets users push frames programmatically.
LiveKitSource is pass-through (user manages their own room).
"""

from __future__ import annotations

import asyncio
import logging
import platform
import time
from typing import Any, Optional

from livekit import rtc as livekit_rtc

from ._constants import FFMPEG_MAX_FPS, FFMPEG_MIN_FPS
from ._ffmpeg import FFmpegSource
from .errors import SourceEndedError
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
)

logger = logging.getLogger("overshoot")

# Reduces startup latency and buffering for real-time network sources
_LOW_LATENCY_FLAGS = (
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-analyzeduration", "500000",
    "-probesize", "500000",
)


class ResolvedSource:
    """Result of resolving a SourceConfig.

    Holds the wire-ready source to send to the API, plus optional
    FFmpeg source and LiveKit video track/source that need to be
    kept alive and cleaned up when the stream closes.
    """

    def __init__(
        self,
        wire_source: Optional[WireSource],
        ffmpeg_source: Optional[FFmpegSource] = None,
        livekit_video_source: Optional[Any] = None,
        livekit_video_track: Optional[Any] = None,
        pump_task: Optional[asyncio.Task[None]] = None,
    ) -> None:
        self.wire_source = wire_source
        self.ffmpeg_source = ffmpeg_source
        self.livekit_video_source = livekit_video_source
        self.livekit_video_track = livekit_video_track
        self.pump_task = pump_task

    @property
    def is_native(self) -> bool:
        """True if this source uses native LiveKit transport (Python pump path)."""
        return self.wire_source is None and self.livekit_video_track is not None

    async def close(self) -> None:
        """Clean up all resources."""
        if self.pump_task is not None:
            self.pump_task.cancel()
            try:
                await self.pump_task
            except asyncio.CancelledError:
                pass
            self.pump_task = None
        if self.ffmpeg_source is not None:
            await self.ffmpeg_source.stop()
            self.ffmpeg_source = None
            logger.debug("FFmpeg source stopped")
        if self.livekit_video_source is not None:
            self.livekit_video_source = None
        if self.livekit_video_track is not None:
            self.livekit_video_track = None


def _clamp_fps(target_fps: int) -> int:
    """Clamp target FPS to the valid range."""
    clamped = max(FFMPEG_MIN_FPS, min(target_fps, FFMPEG_MAX_FPS))
    if clamped != target_fps:
        logger.warning("target_fps %d clamped to %d (valid range: %d-%d)",
                       target_fps, clamped, FFMPEG_MIN_FPS, FFMPEG_MAX_FPS)
    return clamped


async def resolve_source(
    source: SourceConfig,
    *,
    target_fps: int = 15,
) -> ResolvedSource:
    """Convert any SourceConfig into a wire-ready source.

    - LiveKitSource -> pass-through (user manages their own room).
    - FrameSource -> creates LiveKit video track, user pushes frames.
    - FileSource / RTSPSource / HLSSource / RTMPSource / CameraSource ->
      FFmpeg decodes frames, creates LiveKit video track.
    """
    if isinstance(source, LiveKitSource):
        return ResolvedSource(wire_source=source)

    if isinstance(source, FrameSource):
        return _resolve_frame_source(source)

    if isinstance(source, FileSource):
        return await _resolve_ffmpeg_source(
            source.path, target_fps,
            loop=source.loop, name=f"file:{source.path}",
        )

    if isinstance(source, RTSPSource):
        return await _resolve_ffmpeg_source(
            source.url, target_fps,
            extra_input_args=[
                *_LOW_LATENCY_FLAGS,
                "-rtsp_transport", "tcp", "-rtsp_flags", "prefer_tcp",
            ],
            name=f"rtsp:{source.url}",
        )

    if isinstance(source, HLSSource):
        return await _resolve_ffmpeg_source(
            source.url, target_fps,
            extra_input_args=list(_LOW_LATENCY_FLAGS),
            name=f"hls:{source.url}",
        )

    if isinstance(source, RTMPSource):
        return await _resolve_ffmpeg_source(
            source.url, target_fps,
            extra_input_args=list(_LOW_LATENCY_FLAGS),
            name=f"rtmp:{source.url}",
        )

    if isinstance(source, CameraSource):
        return await _resolve_camera_source(source, target_fps)

    raise TypeError(f"Unsupported source type: {type(source)}")


# ── FrameSource ──────────────────────────────────────────────────────


def _resolve_frame_source(source: FrameSource) -> ResolvedSource:
    """Create a LiveKit video track for programmatic frame injection."""
    video_source = livekit_rtc.VideoSource(source.width, source.height)
    video_track = livekit_rtc.LocalVideoTrack.create_video_track("video", video_source)

    # Attach LiveKit objects so push_frame() works
    source._livekit_video_source = video_source
    source._livekit_video_track = video_track

    logger.debug("Created FrameSource (%dx%d)", source.width, source.height)

    return ResolvedSource(
        wire_source=None,
        livekit_video_source=video_source,
        livekit_video_track=video_track,
    )


# ── FFmpeg-based sources ─────────────────────────────────────────────


async def _resolve_ffmpeg_source(
    input_path: str,
    target_fps: int,
    *,
    loop: bool = False,
    extra_input_args: Optional[list[str]] = None,
    name: str = "",
) -> ResolvedSource:
    """Create an FFmpeg source and LiveKit video track from a file or URL."""
    ffmpeg_fps = _clamp_fps(target_fps)

    ffmpeg = FFmpegSource(
        input_path,
        target_fps=ffmpeg_fps,
        loop=loop,
        extra_input_args=extra_input_args,
    )
    await ffmpeg.start()

    video_source = livekit_rtc.VideoSource(ffmpeg.width, ffmpeg.height)
    video_track = livekit_rtc.LocalVideoTrack.create_video_track("video", video_source)

    pump_task = asyncio.create_task(
        _pump_frames(ffmpeg, video_source),
        name=f"overshoot-pump-{name[:30]}",
    )

    logger.debug("Created FFmpeg source: %s (%dx%d @ %dfps)",
                 name, ffmpeg.width, ffmpeg.height, ffmpeg_fps)

    return ResolvedSource(
        wire_source=None,
        ffmpeg_source=ffmpeg,
        livekit_video_source=video_source,
        livekit_video_track=video_track,
        pump_task=pump_task,
    )


# ── Camera source ────────────────────────────────────────────────────


async def _resolve_camera_source(source: CameraSource, target_fps: int) -> ResolvedSource:
    """Create an FFmpeg source and LiveKit video track from a camera device."""
    device = source.device
    fmt = None

    if device == "default":
        system = platform.system()
        if system == "Linux":
            device = "/dev/video0"
            fmt = "v4l2"
        elif system == "Darwin":
            device = "0:none"
            fmt = "avfoundation"
        else:
            device = "video=0"
            fmt = "dshow"
    elif device.startswith("/dev/"):
        fmt = "v4l2"

    ffmpeg_fps = _clamp_fps(target_fps)

    # Camera devices require framerate and video_size as input options.
    # Use 30fps capture; the output fps filter downsamples to target_fps.
    # (avfoundation on macOS rejects lower framerates even when listed as supported)
    extra_input_args = [
        "-framerate", "30",
        "-video_size", f"{source.width}x{source.height}",
    ]

    ffmpeg = FFmpegSource(
        device,
        target_fps=ffmpeg_fps,
        width=source.width,
        height=source.height,
        loop=False,
        input_format=fmt,
        extra_input_args=extra_input_args,
        probe=False,  # can't probe camera devices
    )
    await ffmpeg.start()

    video_source = livekit_rtc.VideoSource(ffmpeg.width, ffmpeg.height)
    video_track = livekit_rtc.LocalVideoTrack.create_video_track("video", video_source)

    pump_task = asyncio.create_task(
        _pump_frames(ffmpeg, video_source),
        name="overshoot-pump-camera",
    )

    logger.debug("Created camera source: %s (%dx%d @ %dfps)",
                 source.device, ffmpeg.width, ffmpeg.height, ffmpeg_fps)

    return ResolvedSource(
        wire_source=None,
        ffmpeg_source=ffmpeg,
        livekit_video_source=video_source,
        livekit_video_track=video_track,
        pump_task=pump_task,
    )


# ── Frame pump ───────────────────────────────────────────────────────


async def _pump_frames(
    ffmpeg: FFmpegSource,
    video_source: Any,
) -> None:
    """Read frames from FFmpeg at full source rate and publish every frame
    to LiveKit. Backend handles FPS sampling.

    Raises ``SourceEndedError`` when FFmpeg stops producing frames so that
    the Stream can detect the failure via the task's result.
    """
    published_count = 0
    stats_start = time.monotonic()
    last_stats_time = time.monotonic()

    # Instrumentation
    _read_total_ms = 0.0
    _capture_total_ms = 0.0
    _frames_since_stats = 0
    _read_max_ms = 0.0
    _capture_max_ms = 0.0

    try:
        while True:
            t_before_read = time.monotonic()
            frame_info = await ffmpeg.read_frame()
            t_after_read = time.monotonic()
            read_ms = (t_after_read - t_before_read) * 1000

            if frame_info is None:
                stderr = ffmpeg.last_stderr
                raise SourceEndedError(
                    f"Video source ended unexpectedly (source: {ffmpeg.source})",
                    stderr=stderr or None,
                )

            frame = livekit_rtc.VideoFrame(
                frame_info.width,
                frame_info.height,
                livekit_rtc.VideoBufferType.NV12,
                frame_info.data,
            )
            t_before_capture = time.monotonic()
            video_source.capture_frame(frame)
            t_after_capture = time.monotonic()
            capture_ms = (t_after_capture - t_before_capture) * 1000

            published_count += 1
            _frames_since_stats += 1
            _read_total_ms += read_ms
            _capture_total_ms += capture_ms
            _read_max_ms = max(_read_max_ms, read_ms)
            _capture_max_ms = max(_capture_max_ms, capture_ms)

            # Yield to event loop periodically to prevent starvation
            # under high-fps sources (e.g. 100fps H.265)
            if published_count % 30 == 0:
                await asyncio.sleep(0)

            now = time.monotonic()
            if now - last_stats_time >= 10.0:
                elapsed = now - stats_start
                actual_fps = published_count / elapsed if elapsed > 0 else 0
                n = _frames_since_stats or 1
                logger.info(
                    "FPS stats: published=%.1f frames=%d | "
                    "read avg=%.1fms max=%.1fms | "
                    "capture avg=%.1fms max=%.1fms",
                    actual_fps, published_count,
                    _read_total_ms / n, _read_max_ms,
                    _capture_total_ms / n, _capture_max_ms,
                )
                last_stats_time = now
                _read_total_ms = 0.0
                _capture_total_ms = 0.0
                _frames_since_stats = 0
                _read_max_ms = 0.0
                _capture_max_ms = 0.0
    except asyncio.CancelledError:
        pass
