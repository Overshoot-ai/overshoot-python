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

from ._ffmpeg import FFmpegSource
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
    ) -> None:
        self.wire_source = wire_source
        self.ffmpeg_source = ffmpeg_source
        self.livekit_video_source = livekit_video_source
        self.livekit_video_track = livekit_video_track

    @property
    def is_native(self) -> bool:
        """True if this source uses native LiveKit transport."""
        return self.wire_source is None and self.livekit_video_track is not None

    async def close(self) -> None:
        """Clean up the FFmpeg source and LiveKit references."""
        if self.ffmpeg_source is not None:
            await self.ffmpeg_source.stop()
            self.ffmpeg_source = None
            logger.debug("FFmpeg source stopped")
        if self.livekit_video_source is not None:
            self.livekit_video_source = None
        if self.livekit_video_track is not None:
            self.livekit_video_track = None


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
            extra_input_args=["-rtsp_transport", "tcp", "-rtsp_flags", "prefer_tcp"],
            name=f"rtsp:{source.url}",
        )

    if isinstance(source, HLSSource):
        return await _resolve_ffmpeg_source(
            source.url, target_fps,
            name=f"hls:{source.url}",
        )

    if isinstance(source, RTMPSource):
        return await _resolve_ffmpeg_source(
            source.url, target_fps,
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
    ffmpeg_fps = max(target_fps, 15)

    ffmpeg = FFmpegSource(
        input_path,
        target_fps=ffmpeg_fps,
        loop=loop,
        extra_input_args=extra_input_args,
    )
    await ffmpeg.start()

    video_source = livekit_rtc.VideoSource(ffmpeg.width, ffmpeg.height)
    video_track = livekit_rtc.LocalVideoTrack.create_video_track("video", video_source)

    asyncio.create_task(
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

    ffmpeg_fps = max(target_fps, 15)

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

    asyncio.create_task(
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
    )


# ── Frame pump ───────────────────────────────────────────────────────


async def _pump_frames(
    ffmpeg: FFmpegSource,
    video_source: Any,
) -> None:
    """Read frames from FFmpeg and push to LiveKit with monotonic clock pacing."""
    interval = 1.0 / ffmpeg.target_fps
    next_frame_time = time.monotonic()
    frame_count = 0
    stats_start = time.monotonic()
    last_stats_time = time.monotonic()

    while True:
        frame_info = await ffmpeg.read_frame()
        if frame_info is None:
            logger.warning("FFmpeg source ended")
            break

        frame = livekit_rtc.VideoFrame(
            frame_info.width,
            frame_info.height,
            livekit_rtc.VideoBufferType.RGBA,
            frame_info.data,
        )
        video_source.capture_frame(frame)
        t_capture = time.monotonic()

        frame_count += 1

        next_frame_time += interval
        sleep_for = next_frame_time - t_capture
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

        now = time.monotonic()
        if now - last_stats_time >= 10.0:
            elapsed = now - stats_start
            actual_fps = frame_count / elapsed if elapsed > 0 else 0
            logger.info(
                "FPS stats: target=%.1f actual=%.1f frames=%d",
                ffmpeg.target_fps, actual_fps, frame_count,
            )
            last_stats_time = now
