"""Read video frames from a file (or URL) using an FFmpeg subprocess.

Requires ``ffmpeg`` and ``ffprobe`` on PATH.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from collections import deque
from dataclasses import dataclass
from typing import Optional

from ._constants import FFMPEG_READ_TIMEOUT_SECONDS
from .errors import OvershootError

logger = logging.getLogger("overshoot")

FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"


@dataclass
class FrameInfo:
    width: int
    height: int
    data: bytes  # NV12 raw pixels


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


async def _probe_resolution(source: str) -> tuple[int, int]:
    """Use ffprobe to get video width and height (async, non-blocking)."""
    cmd = [FFPROBE_BIN, "-v", "error"]
    # RTSP sources need TCP transport for reliable probing (especially H.265)
    if source.startswith("rtsp://"):
        cmd += ["-rtsp_transport", "tcp"]
    cmd += [
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        source,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
    if proc.returncode != 0:
        raise OvershootError(f"ffprobe failed: {stderr.decode().strip()}")
    output = stdout.decode().strip()
    parts = output.split("x")
    if len(parts) != 2:
        raise OvershootError(f"ffprobe returned unexpected output: {output}")
    return int(parts[0]), int(parts[1])


class FFmpegSource:
    """Spawns an FFmpeg process that decodes a video source to raw NV12 frames
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
        read_timeout: float = FFMPEG_READ_TIMEOUT_SECONDS,
    ) -> None:
        _check_ffmpeg()

        self.source = source
        self.target_fps = target_fps
        self.loop = loop
        self._input_format = input_format
        self._extra_input_args = extra_input_args or []
        self._read_timeout = read_timeout
        self._probe = probe

        if width and height:
            self.width, self.height = width, height
        else:
            # Will be resolved in start() via async ffprobe if probe=True
            self.width, self.height = 640, 480

        self._width_height_set = width is not None and height is not None
        self.frame_size = 0  # Set in start() after resolution is known
        self._process: Optional[asyncio.subprocess.Process] = None
        self._stderr_task: Optional[asyncio.Task[None]] = None
        self._stderr_lines: deque[str] = deque(maxlen=20)

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
            "-vf", f"scale={self.width}:{self.height}",
            "-pix_fmt", "nv12",
            "-f", "rawvideo",
            "-an",
            "-v", "error",
            "pipe:1",
        ]
        return cmd

    async def start(self) -> None:
        """Probe resolution (if needed) and start the FFmpeg subprocess."""
        # Resolve dimensions via async ffprobe if not explicitly set
        if not self._width_height_set and self._probe:
            self.width, self.height = await _probe_resolution(self.source)

        # Cap resolution to 1280x720 to save bandwidth
        if self.width > 1280 or self.height > 720:
            scale = min(1280 / self.width, 720 / self.height)
            self.width = int(self.width * scale) & ~1  # ensure even
            self.height = int(self.height * scale) & ~1

        self.frame_size = self.width * self.height * 3 // 2  # NV12

        cmd = self._build_cmd()
        logger.info("Starting FFmpeg: %s", " ".join(cmd))
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # Drain stderr to prevent pipe buffer deadlock
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(),
            name="overshoot-ffmpeg-stderr",
        )

    async def _drain_stderr(self) -> None:
        """Read stderr continuously to prevent pipe buffer deadlock."""
        if not self._process or not self._process.stderr:
            return
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    self._stderr_lines.append(text)
                    logger.debug("FFmpeg stderr: %s", text)
        except asyncio.CancelledError:
            pass

    @property
    def last_stderr(self) -> str:
        """Return the last stderr lines from FFmpeg (useful for error reporting)."""
        return "\n".join(self._stderr_lines)

    async def read_frame(self) -> Optional[FrameInfo]:
        """Read a single NV12 frame. Returns None on EOF or error."""
        if not self._process or not self._process.stdout:
            return None

        try:
            data = await asyncio.wait_for(
                self._process.stdout.readexactly(self.frame_size),
                timeout=self._read_timeout,
            )
            return FrameInfo(width=self.width, height=self.height, data=data)
        except asyncio.IncompleteReadError:
            return None
        except asyncio.TimeoutError:
            logger.error(
                "FFmpeg read timed out after %.0fs (source: %s)",
                self._read_timeout, self.source,
            )
            return None

    async def stop(self) -> None:
        """Terminate the FFmpeg subprocess."""
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
            self._stderr_task = None

        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except (ProcessLookupError, asyncio.TimeoutError):
                self._process.kill()
            self._process = None
