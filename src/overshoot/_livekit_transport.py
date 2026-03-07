"""Native LiveKit transport — connects to a server-managed LiveKit room and publishes video.

Requires the ``livekit`` package (``pip install overshoot[livekit]``).
"""

import logging
from typing import Any, Callable, Optional

from livekit import rtc as livekit_rtc

from .errors import OvershootError

logger = logging.getLogger("overshoot")


class LiveKitTransport:
    """Manages a LiveKit room connection for publishing local video.

    The server creates the room; this class connects as a client and
    publishes a single video track.
    """

    def __init__(
        self,
        *,
        on_fatal_error: Optional[Callable[[Exception], Any]] = None,
    ) -> None:
        self._room: Optional[livekit_rtc.Room] = None
        self._token: Optional[str] = None
        self._on_fatal_error = on_fatal_error
        self._connected = False

    async def connect(
        self,
        url: str,
        token: str,
        video_track: Any,
    ) -> None:
        """Connect to a LiveKit room and publish the video track."""
        self._token = token

        room = livekit_rtc.Room()

        @room.on("reconnecting")
        def _on_reconnecting() -> None:
            logger.warning("LiveKit room reconnecting...")

        @room.on("reconnected")
        def _on_reconnected() -> None:
            logger.info("LiveKit room reconnected")

        @room.on("disconnected")
        def _on_disconnected(reason: Any = None) -> None:
            if not self._connected:
                return
            logger.error("LiveKit room disconnected: %s", reason)
            if self._on_fatal_error is not None:
                self._on_fatal_error(
                    OvershootError(f"LiveKit room disconnected: {reason}")
                )

        # single_peer_connection=True is required for the Rust WebRTC stack
        opts = livekit_rtc.RoomOptions(single_peer_connection=True)
        await room.connect(url, token, options=opts)
        self._room = room
        self._connected = True
        logger.info("Connected to LiveKit room %s", room.name)

        # Publish video track
        options = livekit_rtc.TrackPublishOptions(
            source=livekit_rtc.TrackSource.SOURCE_CAMERA,
        )
        publication = await room.local_participant.publish_track(video_track, options)
        logger.info("Published video track: %s", publication.sid)

    def update_token(self, token: str) -> None:
        """Store a fresh token from keepalive for potential reconnection."""
        self._token = token
        logger.debug("LiveKit token updated")

    async def disconnect(self) -> None:
        """Disconnect from the LiveKit room."""
        self._connected = False
        if self._room is not None:
            await self._room.disconnect()
            self._room = None
            logger.info("Disconnected from LiveKit room")
