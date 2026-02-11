import asyncio
import logging
from typing import Any, Optional

import aiohttp

from .errors import (
    ApiError,
    AuthenticationError,
    InsufficientCreditsError,
    NetworkError,
    NotFoundError,
    ServerError,
    ValidationError,
)

logger = logging.getLogger("overshoot")


class HttpClient:
    """Internal HTTP helper shared by Overshoot and ApiClient.

    Manages an aiohttp session, adds auth headers, and maps HTTP
    status codes to SDK exceptions.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def base_url(self) -> str:
        return self._base_url

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
            )
        return self._session

    async def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Make an HTTP request and return the parsed JSON response.

        Raises SDK-specific exceptions for error status codes.
        """
        session = await self._ensure_session()
        url = f"{self._base_url}{path}"

        try:
            async with session.request(method, url, json=json_body) as resp:
                if resp.status == 204:
                    return {"status": "ok"}

                try:
                    data: dict[str, Any] = await resp.json()
                except Exception:
                    data = {"error": "unknown", "message": await resp.text()}

                if resp.ok:
                    return data

                message = data.get("message") or data.get("error", "Unknown error")
                request_id = data.get("request_id")
                details = data.get("details")

                if resp.status == 401:
                    raise AuthenticationError(message, request_id=request_id)
                if resp.status == 402:
                    raise InsufficientCreditsError(message, request_id=request_id, details=details)
                if resp.status in (400, 422):
                    raise ValidationError(
                        message, status_code=resp.status, request_id=request_id, details=details
                    )
                if resp.status == 404:
                    raise NotFoundError(message, request_id=request_id)
                if resp.status == 429:
                    raise ApiError(
                        message, status_code=429, request_id=request_id, details=details
                    )
                if resp.status >= 500:
                    raise ServerError(
                        message, status_code=resp.status, request_id=request_id, details=details
                    )
                raise ApiError(
                    message, status_code=resp.status, request_id=request_id, details=details
                )

        except aiohttp.ClientError as exc:
            raise NetworkError(f"Network error: {exc}", cause=exc) from exc
        except asyncio.TimeoutError as exc:
            raise NetworkError("Request timed out", cause=exc) from exc

    def ws_url(self, stream_id: str) -> str:
        """Build the WebSocket URL for a given stream."""
        ws_base = self._base_url.replace("https://", "wss://").replace("http://", "ws://")
        return f"{ws_base}/ws/streams/{stream_id}"

    async def ws_connect(self, url: str) -> aiohttp.ClientWebSocketResponse:
        """Open a WebSocket connection."""
        session = await self._ensure_session()
        return await session.ws_connect(url)

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
