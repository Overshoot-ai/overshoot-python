"""Read video frames from a file (or URL) using an FFmpeg subprocess.

Requires ``ffmpeg`` and ``ffprobe`` on PATH.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from dataclasses import dataclass
from typing import Optional

from .errors import OvershootError

logger = logging.getLogger("overshoot")

FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"


@dataclass
class FrameInfo:
    width: int
    height: int
    data: bytes  # RGBA raw pixels


def _check_ffmpeg() -> None:
    """Verify that ffmpeg and ffprobe are available on PATH."""
    for binary in (FFMPEG_BIN, FFPROBE_BIN):
        try:
            subprocess.run(
                [binary, "-version"],
                capture_output=True,
                timeout=5,
            )
        except FileNotFoundError:
            raise OvershootError(
                f"'{binary}' not found on PATH. "
                "Install FFmpeg: https://ffmpeg.org/download.html"
            )


def _probe_resolution(source: str) -> tuple[int, int]:
    """Use ffprobe to get video width and height."""
    cmd = [
        FFPROBE_BIN, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        source,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        raise OvershootError(f"ffprobe failed: {result.stderr.strip()}")
    parts = result.stdout.strip().split("x")
    if len(parts) != 2:
        raise OvershootError(f"ffprobe returned unexpected output: {result.stdout.strip()}")
    return int(parts[0]), int(parts[1])


class FFmpegSource:
    """Spawns an FFmpeg process that decodes a video source to raw RGBA frames
    piped to stdout.

    Supports local files, HLS URLs, RTSP, RTMP, etc.
    For local files, the video loops indefinitely (``-stream_loop -1``).
    """

    def __init__(
        self,
        source: str,
        *,
        target_fps: int = 15,
        width: Optional[int] = None,
        height: Optional[int] = None,
        loop: bool = True,
        input_format: Optional[str] = None,
        extra_input_args: Optional[list[str]] = None,
        probe: bool = True,
    ) -> None:
        _check_ffmpeg()

        self.source = source
        self.target_fps = target_fps
        self.loop = loop
        self._input_format = input_format
        self._extra_input_args = extra_input_args or []

        if width and height:
            self.width, self.height = width, height
        elif probe:
            self.width, self.height = _probe_resolution(source)
        else:
            self.width, self.height = 640, 480

        # Cap resolution to 1280x720 to save bandwidth
        if self.width > 1280 or self.height > 720:
            scale = min(1280 / self.width, 720 / self.height)
            self.width = int(self.width * scale) & ~1  # ensure even
            self.height = int(self.height * scale) & ~1

        self.frame_size = self.width * self.height * 4  # RGBA
        self._process: Optional[asyncio.subprocess.Process] = None

    def _build_cmd(self) -> list[str]:
        cmd = [FFMPEG_BIN]

        # Input format (e.g. avfoundation, v4l2, dshow)
        if self._input_format:
            cmd += ["-f", self._input_format]

        # Extra input args (e.g. RTSP TCP transport flags)
        if self._extra_input_args:
            cmd += self._extra_input_args

        # Loop for local files (not for network streams or devices)
        is_network = self.source.startswith(("http://", "https://", "rtsp://", "rtmp://"))
        if self.loop and not is_network and not self._input_format:
            cmd += ["-stream_loop", "-1"]

        cmd += [
            "-i", self.source,
            "-vf", f"fps={self.target_fps},scale={self.width}:{self.height}",
            "-pix_fmt", "rgba",
            "-f", "rawvideo",
            "-an",
            "-v", "error",
            "pipe:1",
        ]
        return cmd

    async def start(self) -> None:
        """Start the FFmpeg subprocess."""
        cmd = self._build_cmd()
        logger.info("Starting FFmpeg: %s", " ".join(cmd))
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def read_frame(self) -> Optional[FrameInfo]:
        """Read a single RGBA frame. Returns None on EOF or error."""
        if not self._process or not self._process.stdout:
            return None

        data = b""
        remaining = self.frame_size
        while remaining > 0:
            chunk = await self._process.stdout.read(remaining)
            if not chunk:
                return None
            data += chunk
            remaining -= len(chunk)

        return FrameInfo(width=self.width, height=self.height, data=data)

    async def stop(self) -> None:
        """Terminate the FFmpeg subprocess."""
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except (ProcessLookupError, asyncio.TimeoutError):
                self._process.kill()
            self._process = None
