"""Source resolution — converts FileSource/CameraSource into publishable media.

When the ``livekit`` package is installed (``pip install overshoot[livekit]``),
local sources default to native LiveKit transport (server creates the room).
Otherwise, falls back to WebRTC via ``aiortc`` (``pip install overshoot[webrtc]``).

LiveKitSource, WebRTCSource, and NativeSource are pass-through and need no
extra dependencies.
"""

import logging
from typing import Any, Literal, Optional

TransportType = Literal["auto", "livekit", "webrtc"]

from .errors import OvershootError
from .types import (
    CameraSource,
    FileSource,
    LiveKitSource,
    NativeSource,
    SourceConfig,
    WebRTCAnswer,
    WebRTCSource,
    WireSource,
)

logger = logging.getLogger("overshoot")

# ── Optional dependency: livekit ─────────────────────────────────────

try:
    from livekit import rtc as livekit_rtc

    HAS_LIVEKIT = True
except ImportError:
    HAS_LIVEKIT = False
    livekit_rtc = None  # type: ignore[assignment]

# ── Optional dependency: aiortc (legacy WebRTC) ─────────────────────

try:
    from aiortc import (
        RTCPeerConnection,
        RTCConfiguration,
        RTCIceServer,
        RTCSessionDescription,
    )
    from aiortc.contrib.media import MediaPlayer

    HAS_AIORTC = True
except ImportError:
    HAS_AIORTC = False
    RTCPeerConnection = Any  # type: ignore[assignment,misc]
    MediaPlayer = Any  # type: ignore[assignment,misc]


def _require_aiortc(source_type: str) -> None:
    if not HAS_AIORTC:
        raise OvershootError(
            f"{source_type} requires aiortc. Install with: pip install overshoot[webrtc]"
        )


class ResolvedSource:
    """Result of resolving a SourceConfig.

    Holds the wire-ready source to send to the API, plus optional
    RTCPeerConnection and MediaPlayer that need to be kept alive
    and cleaned up when the stream closes.

    For native LiveKit transport, ``wire_source`` is None and
    ``livekit_video_track`` holds the track to publish.
    """

    def __init__(
        self,
        wire_source: Optional[WireSource],
        peer_connection: Optional[Any] = None,
        media_player: Optional[Any] = None,
        livekit_video_source: Optional[Any] = None,
        livekit_video_track: Optional[Any] = None,
    ) -> None:
        self.wire_source = wire_source
        self.peer_connection = peer_connection
        self.media_player = media_player
        self.livekit_video_source = livekit_video_source
        self.livekit_video_track = livekit_video_track

    @property
    def is_native(self) -> bool:
        """True if this source uses native LiveKit transport."""
        return self.wire_source is None and self.livekit_video_track is not None

    async def apply_answer(self, answer: WebRTCAnswer) -> None:
        """Set the server's SDP answer on the peer connection."""
        if self.peer_connection is not None:
            await self.peer_connection.setRemoteDescription(
                RTCSessionDescription(sdp=answer.sdp, type=answer.type)
            )
            logger.debug("Applied SDP answer to peer connection")

    async def close(self) -> None:
        """Clean up the peer connection, media player, and LiveKit source."""
        if self.peer_connection is not None:
            await self.peer_connection.close()
            self.peer_connection = None
            logger.debug("Peer connection closed")
        if self.media_player is not None:
            self.media_player = None
            logger.debug("Media player stopped")
        if self.livekit_video_source is not None:
            self.livekit_video_source = None
        if self.livekit_video_track is not None:
            self.livekit_video_track = None


async def resolve_source(
    source: SourceConfig,
    ice_servers: Optional[list[Any]] = None,
    transport: TransportType = "auto",
) -> ResolvedSource:
    """Convert any SourceConfig into a wire-ready source.

    - NativeSource → returned as-is (wire_source=None).
    - LiveKitSource / WebRTCSource → returned as-is.
    - FileSource / CameraSource:
      - ``transport="auto"`` (default): uses LiveKit if installed, else aiortc.
      - ``transport="livekit"``: forces native LiveKit (raises if not installed).
      - ``transport="webrtc"``: forces aiortc WebRTC (raises if not installed).
    """
    if isinstance(source, NativeSource):
        return ResolvedSource(wire_source=None)

    if isinstance(source, (LiveKitSource, WebRTCSource)):
        return ResolvedSource(wire_source=source)

    use_livekit = (
        (transport == "livekit")
        or (transport == "auto" and HAS_LIVEKIT)
    )

    if transport == "livekit" and not HAS_LIVEKIT:
        raise OvershootError(
            "transport='livekit' requires the livekit package. "
            "Install with: pip install overshoot[livekit]"
        )
    if transport == "webrtc" and not HAS_AIORTC:
        raise OvershootError(
            "transport='webrtc' requires aiortc. "
            "Install with: pip install overshoot[webrtc]"
        )

    if isinstance(source, FileSource):
        if use_livekit:
            return await _resolve_file_source_livekit(source)
        _require_aiortc("FileSource")
        return await _resolve_file_source_webrtc(source, ice_servers)

    if isinstance(source, CameraSource):
        if use_livekit:
            return await _resolve_camera_source_livekit(source)
        _require_aiortc("CameraSource")
        return await _resolve_camera_source_webrtc(source, ice_servers)

    raise TypeError(f"Unsupported source type: {type(source)}")


# ── Native LiveKit source resolution ────────────────────────────────


def _capture_one_frame(container: Any, stream: Any, video_source: Any) -> None:
    """Decode one frame and push it to the video source. Ensures the track is producing."""
    for frame in container.decode(stream):
        _push_frame(frame, video_source)
        return
    raise OvershootError("Could not decode any frames from source")


def _push_frame(frame: Any, video_source: Any) -> None:
    """Convert an av frame to a LiveKit VideoFrame and capture it."""
    img = frame.to_ndarray(format="rgba")
    h, w = img.shape[:2]
    lk_frame = livekit_rtc.VideoFrame(
        width=w,
        height=h,
        type=livekit_rtc.VideoBufferType.RGBA,
        data=img.tobytes(),
    )
    video_source.capture_frame(lk_frame)


async def _resolve_file_source_livekit(source: FileSource) -> ResolvedSource:
    """Create a LiveKit video source from a local video file."""
    import asyncio

    import av  # type: ignore[import-untyped]

    container = av.open(source.path)
    stream = container.streams.video[0]

    width = stream.width or 640
    height = stream.height or 480

    video_source = livekit_rtc.VideoSource(width=width, height=height)
    video_track = livekit_rtc.LocalVideoTrack.create_video_track(
        "video", video_source
    )

    # Push the first frame immediately so the track is "ready" before publish
    _capture_one_frame(container, stream, video_source)
    container.seek(0)

    async def _pump_frames() -> None:
        """Read video frames and push them to the LiveKit video source."""
        fps = float(stream.average_rate or 30)
        frame_interval = 1.0 / fps

        while True:
            for frame in container.decode(stream):
                _push_frame(frame, video_source)
                await asyncio.sleep(frame_interval)

            if not source.loop:
                break
            container.seek(0)

    asyncio.create_task(_pump_frames(), name="overshoot-file-pump")

    logger.debug("Created LiveKit video source from file: %s", source.path)

    return ResolvedSource(
        wire_source=None,
        livekit_video_source=video_source,
        livekit_video_track=video_track,
    )


async def _resolve_camera_source_livekit(source: CameraSource) -> ResolvedSource:
    """Create a LiveKit video source from a local camera."""
    import asyncio

    import av  # type: ignore[import-untyped]

    device = source.device
    fmt = None

    if device == "default":
        import platform

        system = platform.system()
        if system == "Linux":
            device = "/dev/video0"
            fmt = "v4l2"
        elif system == "Darwin":
            device = "default"
            fmt = "avfoundation"
        else:
            device = "video=0"
            fmt = "dshow"
    elif device.startswith("/dev/"):
        fmt = "v4l2"

    options: dict[str, str] = {}
    if fmt == "avfoundation":
        options["pixel_format"] = "uyvy422"

    container = av.open(device, format=fmt, options=options)
    stream = container.streams.video[0]

    video_source = livekit_rtc.VideoSource(
        width=stream.width or 640,
        height=stream.height or 480,
    )
    video_track = livekit_rtc.LocalVideoTrack.create_video_track(
        "video", video_source
    )

    # Push the first frame immediately so the track is "ready" before publish
    _capture_one_frame(container, stream, video_source)

    async def _pump_frames() -> None:
        """Read camera frames and push them to the LiveKit video source."""
        fps = float(stream.average_rate or 30)
        frame_interval = 1.0 / fps

        for frame in container.decode(stream):
            _push_frame(frame, video_source)
            await asyncio.sleep(frame_interval)

    asyncio.create_task(_pump_frames(), name="overshoot-camera-pump")

    logger.debug("Created LiveKit video source from camera: %s", source.device)

    return ResolvedSource(
        wire_source=None,
        livekit_video_source=video_source,
        livekit_video_track=video_track,
    )


# ── Legacy WebRTC source resolution (aiortc) ────────────────────────


async def _resolve_file_source_webrtc(
    source: FileSource,
    ice_servers: Optional[list[Any]],
) -> ResolvedSource:
    """Create a WebRTC peer connection streaming from a video file."""
    options: dict[str, str] = {}
    if source.loop:
        options["loop"] = "1"

    player = MediaPlayer(source.path, options=options)

    if player.video is None:
        raise OvershootError(f"No video track found in file: {source.path}")

    pc = _create_peer_connection(ice_servers)
    pc.addTrack(player.video)

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    logger.debug("Created WebRTC offer from file: %s", source.path)

    return ResolvedSource(
        wire_source=WebRTCSource(sdp=pc.localDescription.sdp),
        peer_connection=pc,
        media_player=player,
    )


async def _resolve_camera_source_webrtc(
    source: CameraSource,
    ice_servers: Optional[list[Any]],
) -> ResolvedSource:
    """Create a WebRTC peer connection capturing from a local camera."""
    device = source.device
    fmt = None

    if device == "default":
        import platform

        system = platform.system()
        if system == "Linux":
            device = "/dev/video0"
            fmt = "v4l2"
        elif system == "Darwin":
            device = "default"
            fmt = "avfoundation"
        else:
            device = "video=0"
            fmt = "dshow"

    elif device.startswith("/dev/"):
        fmt = "v4l2"

    player = MediaPlayer(device, format=fmt)

    if player.video is None:
        raise OvershootError(f"No video track from camera device: {source.device}")

    pc = _create_peer_connection(ice_servers)
    pc.addTrack(player.video)

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    logger.debug("Created WebRTC offer from camera: %s", source.device)

    return ResolvedSource(
        wire_source=WebRTCSource(sdp=pc.localDescription.sdp),
        peer_connection=pc,
        media_player=player,
    )


def _create_peer_connection(ice_servers: Optional[list[Any]] = None) -> Any:
    """Create an RTCPeerConnection with optional ICE servers."""
    config_kwargs: dict[str, Any] = {}
    if ice_servers:
        config_kwargs["iceServers"] = [
            RTCIceServer(urls=s.urls, username=s.username, credential=s.credential)
            for s in ice_servers
        ]

    config = RTCConfiguration(**config_kwargs)
    return RTCPeerConnection(configuration=config)
