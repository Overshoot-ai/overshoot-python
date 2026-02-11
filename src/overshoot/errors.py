from typing import Any, Optional


class OvershootError(Exception):
    """Base exception for all Overshoot SDK errors."""


class ApiError(OvershootError):
    """HTTP API error with status code and optional server details."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.details = details

    def __repr__(self) -> str:
        parts = [f"ApiError({self.args[0]!r}"]
        if self.status_code is not None:
            parts.append(f", status_code={self.status_code}")
        if self.request_id is not None:
            parts.append(f", request_id={self.request_id!r}")
        parts.append(")")
        return "".join(parts)


class AuthenticationError(ApiError):
    """401 Unauthorized — invalid or revoked API key."""

    def __init__(self, message: str = "Invalid or revoked API key", **kwargs: Any) -> None:
        super().__init__(message, status_code=401, **kwargs)


class ValidationError(ApiError):
    """400/422 — invalid request parameters."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        status_code = kwargs.pop("status_code", 422)
        super().__init__(message, status_code=status_code, **kwargs)


class NotFoundError(ApiError):
    """404 — stream or resource not found."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, status_code=404, **kwargs)


class InsufficientCreditsError(ApiError):
    """402 — not enough credits to create or maintain a stream."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, status_code=402, **kwargs)


class ServerError(ApiError):
    """5xx — server-side error."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        status_code = kwargs.pop("status_code", 500)
        super().__init__(message, status_code=status_code, **kwargs)


class NetworkError(OvershootError):
    """Connection or transport-level failure (DNS, timeout, socket reset)."""

    def __init__(self, message: str, *, cause: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.__cause__ = cause


class StreamClosedError(OvershootError):
    """Operation attempted on a stream that is already closed."""


class WebSocketError(OvershootError):
    """WebSocket connection or protocol error."""

    def __init__(self, message: str, *, code: Optional[int] = None) -> None:
        super().__init__(message)
        self.code = code
