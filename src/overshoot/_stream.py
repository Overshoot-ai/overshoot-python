import asyncio
import json
import logging
from typing import Any, Callable, Optional, TYPE_CHECKING

import aiohttp

from ._constants import (
    WS_RECONNECT_BASE_SECONDS,
    WS_RECONNECT_MAX_SECONDS,
    WS_RECONNECT_MAX_ATTEMPTS,
)
from ._http import HttpClient
from ._sources import ResolvedSource
from .errors import StreamClosedError, WebSocketError
from .types import StreamConfigResponse, StreamInferenceResult

if TYPE_CHECKING:
    from ._livekit_transport import LiveKitTransport

logger = logging.getLogger("overshoot")


class Stream:
    """A running Overshoot analysis stream.

    Created by :meth:`client.streams.create() <overshoot.Overshoot>`,
    not instantiated directly.

    Background tasks consume WebSocket results and renew the lease.
    Call :meth:`close` to stop everything and release resources.
    """

    def __init__(
        self,
        *,
        stream_id: str,
        http: HttpClient,
        resolved_source: ResolvedSource,
        ttl_seconds: int,
        on_result: Callable[[StreamInferenceResult], Any],
        on_error: Optional[Callable[[Exception], Any]],
        livekit_transport: Optional["LiveKitTransport"] = None,
    ) -> None:
        self._stream_id = stream_id
        self._http = http
        self._resolved_source = resolved_source
        self._on_result = on_result
        self._on_error = on_error
        self._ttl_seconds = ttl_seconds
        self._livekit_transport = livekit_transport

        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task[None]] = None
        self._keepalive_task: Optional[asyncio.Task[None]] = None
        self._closed = False

        # WS reconnection state
        self._ws_reconnect_attempts = 0

    # ── Properties ───────────────────────────────────────────────────

    @property
    def stream_id(self) -> str:
        """The server-assigned stream ID."""
        return self._stream_id

    @property
    def is_active(self) -> bool:
        """True if the stream is running and not closed."""
        return not self._closed

    # ── Lifecycle ────────────────────────────────────────────────────

    def _start(self) -> None:
        """Start background tasks. Called by StreamsAPI after construction."""
        self._ws_task = asyncio.create_task(
            self._ws_loop(), name=f"overshoot-ws-{self._stream_id[:8]}"
        )
        if self._ttl_seconds > 0:
            self._keepalive_task = asyncio.create_task(
                self._keepalive_loop(), name=f"overshoot-ka-{self._stream_id[:8]}"
            )

    async def close(self) -> None:
        """Stop all background tasks and release resources.

        Closes the WebSocket, stops keepalive, disconnects LiveKit (if any),
        shuts down the peer connection, and DELETEs the stream on the server
        to trigger final billing. Safe to call multiple times.
        """
        if self._closed:
            return
        # Set closed FIRST to prevent cascade from WS/LiveKit close events
        self._closed = True
        logger.info("Closing stream: %s", self._stream_id)

        # Cancel background tasks
        for task in (self._keepalive_task, self._ws_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._keepalive_task = None
        self._ws_task = None

        # Close WebSocket
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()
            self._ws = None

        # Disconnect LiveKit BEFORE server DELETE (order matters)
        if self._livekit_transport is not None:
            await self._livekit_transport.disconnect()
            self._livekit_transport = None

        # Close peer connection / media player
        await self._resolved_source.close()

        # Tell the server to close (triggers final billing)
        try:
            await self._http.request("DELETE", f"/streams/{self._stream_id}")
        except Exception as exc:
            logger.warning("Failed to close stream on server: %s", exc)

    # ── Runtime control ──────────────────────────────────────────────

    async def update_prompt(self, prompt: str) -> StreamConfigResponse:
        """Update the inference prompt while the stream is running."""
        if self._closed:
            raise StreamClosedError("Stream is closed")
        data = await self._http.request(
            "PATCH",
            f"/streams/{self._stream_id}/config/prompt",
            json_body={"prompt": prompt},
        )
        return StreamConfigResponse(
            id=data["id"],
            stream_id=data["stream_id"],
            prompt=data["prompt"],
            backend=data["backend"],
            model=data["model"],
            output_schema_json=data.get("output_schema_json"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    # ── Background: WebSocket consumer with auto-reconnect ───────────

    async def _ws_loop(self) -> None:
        """Connect to the WebSocket and deliver results via on_result.

        On unexpected disconnect, reconnects with exponential backoff.
        Fatal close codes (1008 auth, 1001 with reason) stop immediately.
        """
        while not self._closed:
            try:
                await self._ws_connect_and_consume()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._closed:
                    return
                logger.warning("WebSocket connection error: %s", exc)

            # If we get here, the WS disconnected. Decide: reconnect or fatal.
            if self._closed:
                return

            if self._ws_reconnect_attempts >= WS_RECONNECT_MAX_ATTEMPTS:
                self._emit_error(
                    WebSocketError(
                        f"WebSocket reconnection failed after {WS_RECONNECT_MAX_ATTEMPTS} attempts"
                    )
                )
                asyncio.create_task(self.close(), name="overshoot-close-on-ws-fail")
                return

            delay = min(
                WS_RECONNECT_BASE_SECONDS * (2 ** self._ws_reconnect_attempts),
                WS_RECONNECT_MAX_SECONDS,
            )
            self._ws_reconnect_attempts += 1
            logger.info(
                "WebSocket reconnecting (attempt %d/%d) in %.1fs...",
                self._ws_reconnect_attempts,
                WS_RECONNECT_MAX_ATTEMPTS,
                delay,
            )
            await asyncio.sleep(delay)

    async def _ws_connect_and_consume(self) -> None:
        """Single WS connection lifecycle: connect, auth, consume messages."""
        ws_url = self._http.ws_url(self._stream_id)

        # Close stale WS reference
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()

        self._ws = await self._http.ws_connect(ws_url)
        await self._ws.send_json({"api_key": self._http.api_key})
        logger.debug("WebSocket connected: %s", ws_url)

        async for msg in self._ws:
            if self._closed:
                return

            if msg.type == aiohttp.WSMsgType.TEXT:
                self._handle_ws_message(msg.data)
                # Reset reconnect counter on successful message
                self._ws_reconnect_attempts = 0

            elif msg.type == aiohttp.WSMsgType.ERROR:
                # Don't treat as fatal — let the outer loop handle reconnect
                logger.warning("WebSocket error: %s", self._ws.exception())
                return

            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING,
                              aiohttp.WSMsgType.CLOSED):
                if self._closed:
                    return
                if msg.data == 1008:
                    # Auth failure — fatal, no reconnect
                    self._emit_error(
                        WebSocketError("WebSocket auth failed: invalid API key", code=1008)
                    )
                    asyncio.create_task(self.close(), name="overshoot-close-on-ws-auth")
                    return
                if msg.data == 1001 and msg.extra:
                    # Server-initiated close with reason — fatal
                    reason = str(msg.extra) if msg.extra else "server closed connection"
                    self._emit_error(
                        WebSocketError(f"Server closed WebSocket: {reason}", code=1001)
                    )
                    asyncio.create_task(self.close(), name="overshoot-close-on-ws-server")
                    return
                # Other close codes — let outer loop attempt reconnect
                logger.info("WebSocket closed with code %s, will attempt reconnect", msg.data)
                return

    def _handle_ws_message(self, raw: str) -> None:
        """Parse a WebSocket text message and call on_result."""
        try:
            data = json.loads(raw)
            result = StreamInferenceResult(
                id=data["id"],
                stream_id=data["stream_id"],
                mode=data["mode"],
                model_backend=data["model_backend"],
                model_name=data["model_name"],
                prompt=data["prompt"],
                result=data["result"],
                inference_latency_ms=data["inference_latency_ms"],
                total_latency_ms=data["total_latency_ms"],
                ok=data["ok"],
                error=data.get("error"),
                finish_reason=data.get("finish_reason"),
            )
            self._on_result(result)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Malformed WebSocket message: %s", exc)

    # ── Background: keepalive ────────────────────────────────────────

    async def _keepalive_loop(self) -> None:
        """Periodically renew the stream lease."""
        interval = self._ttl_seconds / 2
        try:
            while not self._closed:
                await asyncio.sleep(interval)
                if self._closed:
                    break
                try:
                    data = await self._http.request(
                        "POST", f"/streams/{self._stream_id}/keepalive"
                    )
                    logger.debug("Lease renewed for stream %s", self._stream_id)

                    # Refresh LiveKit token if present
                    livekit_token = data.get("livekit_token")
                    if livekit_token and self._livekit_transport is not None:
                        self._livekit_transport.update_token(livekit_token)

                except Exception as exc:
                    logger.error("Keepalive failed: %s", exc)
                    self._emit_error(exc)  # type: ignore[arg-type]
                    asyncio.create_task(self.close(), name="overshoot-close-on-ka-fail")
                    return
        except asyncio.CancelledError:
            raise

    # ── Internal ─────────────────────────────────────────────────────

    def _emit_error(self, exc: Exception) -> None:
        """Forward an error to the on_error callback if set."""
        if self._on_error is not None:
            self._on_error(exc)
        else:
            logger.error("Stream error (no on_error handler): %s", exc)
