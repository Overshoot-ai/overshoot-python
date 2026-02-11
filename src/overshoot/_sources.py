"""Source resolution — converts FileSource/CameraSource into WebRTC peer connections.

FileSource and CameraSource require ``pip install overshoot[webrtc]`` (aiortc).
LiveKitSource and WebRTCSource are pass-through and need no extra dependencies.
"""
import logging
from typing import Any, Optional

from .errors import OvershootError
from .types import (
    CameraSource,
    FileSource,
    LiveKitSource,
    SourceConfig,
    WebRTCAnswer,
    WebRTCSource,
    WireSource,
)

logger = logging.getLogger("overshoot")

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
    """

    def __init__(
        self,
        wire_source: WireSource,
        peer_connection: Optional[Any] = None,
        media_player: Optional[Any] = None,
    ) -> None:
        self.wire_source = wire_source
        self.peer_connection = peer_connection
        self.media_player = media_player

    async def apply_answer(self, answer: WebRTCAnswer) -> None:
        """Set the server's SDP answer on the peer connection."""
        if self.peer_connection is not None:
            await self.peer_connection.setRemoteDescription(
                RTCSessionDescription(sdp=answer.sdp, type=answer.type)
            )
            logger.debug("Applied SDP answer to peer connection")

    async def close(self) -> None:
        """Clean up the peer connection and media player."""
        if self.peer_connection is not None:
            await self.peer_connection.close()
            self.peer_connection = None
            logger.debug("Peer connection closed")
        if self.media_player is not None:
            self.media_player.stop()
            self.media_player = None
            logger.debug("Media player stopped")


async def resolve_source(
    source: SourceConfig,
    ice_servers: Optional[list[Any]] = None,
) -> ResolvedSource:
    """Convert any SourceConfig into a wire-ready source.

    - LiveKitSource / WebRTCSource → returned as-is.
    - FileSource / CameraSource → creates an RTCPeerConnection, adds
      the video track, generates an SDP offer, and returns the offer
      as a WebRTCSource along with the peer connection.
    """
    if isinstance(source, (LiveKitSource, WebRTCSource)):
        return ResolvedSource(wire_source=source)

    if isinstance(source, FileSource):
        _require_aiortc("FileSource")
        return await _resolve_file_source(source, ice_servers)

    if isinstance(source, CameraSource):
        _require_aiortc("CameraSource")
        return await _resolve_camera_source(source, ice_servers)

    raise TypeError(f"Unsupported source type: {type(source)}")


async def _resolve_file_source(
    source: FileSource,
    ice_servers: Optional[list[Any]],
) -> ResolvedSource:
    """Create a WebRTC peer connection streaming from a video file."""
    options: dict[str, str] = {}
    if source.loop:
        options["loop"] = "1"

    player = MediaPlayer(source.path, options=options)

    if player.video is None:
        player.stop()
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


async def _resolve_camera_source(
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
        player.stop()
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
