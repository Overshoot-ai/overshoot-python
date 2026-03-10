"""Optional Go-based H.264 publisher for high-performance video transport.

When the ``overshoot-publisher`` binary is available on PATH, FFmpeg-based
sources can pipe compressed H.264 directly to LiveKit without decoding to
raw pixels in Python. This eliminates the double codec cycle and reduces
pipe bandwidth by ~30x compared to the raw NV12 path.

The binary is optional — if unavailable, the SDK falls back to the standard
FFmpeg → NV12 → Python → LiveKit pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from typing import Optional

from .errors import OvershootError, SourceEndedError

logger = logging.getLogger("overshoot")

PUBLISHER_BIN = "overshoot-publisher"


def go_publisher_available() -> bool:
    """Check if the overshoot-publisher binary is on PATH."""
    return shutil.which(PUBLISHER_BIN) is not None


async def _probe_codec(source: str) -> Optional[str]:
    """Detect the video codec of a source using ffprobe.

    Returns the codec name (e.g. "h264", "hevc", "vp9") or None on failure.
    """
    cmd = ["ffprobe", "-v", "error"]
    # RTSP sources need TCP transport for reliable probing (especially H.265)
    if source.startswith("rtsp://"):
        cmd += ["-rtsp_transport", "tcp"]
    cmd += [
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "csv=p=0",
        source,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        if proc.returncode == 0:
            codec = stdout.decode().strip().split("\n")[0].strip()
            return codec if codec else None
    except (asyncio.TimeoutError, FileNotFoundError):
        pass
    return None


def _build_ffmpeg_h264_cmd(
    source: str,
    *,
    target_fps: int,
    source_codec: Optional[str],
    extra_input_args: Optional[list[str]] = None,
    loop: bool = False,
    input_format: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> list[str]:
    """Build FFmpeg command that outputs H.264 Annex B to stdout."""
    cmd = ["ffmpeg"]

    if input_format:
        cmd += ["-f", input_format]

    if extra_input_args:
        cmd += extra_input_args

    is_network = source.startswith(("http://", "https://", "rtsp://", "rtmp://"))
    if loop and not is_network and not input_format:
        cmd += ["-stream_loop", "-1"]

    cmd += ["-i", source]

    if source_codec == "h264":
        # H.264 source — true passthrough (no filters, no transcode)
        cmd += ["-c:v", "copy"]
    else:
        # Non-H.264 source — transcode to H.264 at full source rate.
        # No fps filter: send all frames and let the backend sample,
        # matching the passthrough path behavior for consistent delivery.
        vf_parts = []
        if width and height:
            vf_parts.append(f"scale={width}:{height}")
        if vf_parts:
            cmd += ["-vf", ",".join(vf_parts)]
        # Force keyframes every 2s so LiveKit SFU can start forwarding quickly.
        # x264 defaults to keyint=250 which at low fps means very long gaps.
        keyint = max(target_fps * 2, 12)
        cmd += [
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-g", str(keyint),
            "-keyint_min", str(keyint),
        ]

    cmd += [
        "-f", "h264",
        "-an",
        "-v", "error",
        "pipe:1",
    ]
    return cmd


class GoPublisherSource:
    """Manages an FFmpeg → overshoot-publisher pipeline.

    FFmpeg decodes/transcodes the source to H.264 Annex B on stdout,
    which is piped directly to the Go publisher binary's stdin.
    The Go binary publishes to LiveKit without Python touching the video data.
    """

    def __init__(
        self,
        source: str,
        *,
        livekit_url: str,
        livekit_token: str,
        target_fps: int,
        source_codec: Optional[str] = None,
        extra_input_args: Optional[list[str]] = None,
        loop: bool = False,
        input_format: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        self.source = source
        self._livekit_url = livekit_url
        self._livekit_token = livekit_token
        self._target_fps = target_fps
        self._source_codec = source_codec
        self._extra_input_args = extra_input_args
        self._loop = loop
        self._input_format = input_format
        self._width = width
        self._height = height

        self._ffmpeg_proc: Optional[asyncio.subprocess.Process] = None
        self._publisher_proc: Optional[asyncio.subprocess.Process] = None
        self._pipe_read_fd: Optional[int] = None
        self._stderr_task: Optional[asyncio.Task[None]] = None
        self._monitor_task: Optional[asyncio.Task[None]] = None
        self._on_ended: Optional[asyncio.Future[SourceEndedError]] = None

    async def start(self) -> None:
        """Start the FFmpeg → Go publisher pipeline."""
        if not go_publisher_available():
            raise OvershootError(
                f"'{PUBLISHER_BIN}' not found on PATH. "
                "Install with: pip install overshoot[fast]"
            )

        import os

        # Build FFmpeg command
        ffmpeg_cmd = _build_ffmpeg_h264_cmd(
            self.source,
            target_fps=self._target_fps,
            source_codec=self._source_codec,
            extra_input_args=self._extra_input_args,
            loop=self._loop,
            input_format=self._input_format,
            width=self._width,
            height=self._height,
        )

        # Build publisher command
        publisher_cmd = [
            PUBLISHER_BIN,
            "--url", self._livekit_url,
            "--token", self._livekit_token,
            "--fps", str(self._target_fps),
        ]

        logger.info("Starting Go publisher pipeline")
        logger.info("  FFmpeg: %s", " ".join(ffmpeg_cmd))
        logger.info("  Publisher: %s", " ".join(publisher_cmd[:3] + ["--token", "<redacted>", "--fps", str(self._target_fps)]))

        # Create OS pipe: FFmpeg writes H.264 → publisher reads it
        read_fd, write_fd = os.pipe()
        self._pipe_read_fd = read_fd

        # Start FFmpeg — stdout writes to the pipe
        self._ffmpeg_proc = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=write_fd,
            stderr=asyncio.subprocess.DEVNULL,
        )
        os.close(write_fd)  # Parent doesn't write; FFmpeg owns it

        # Start publisher — stdin reads from the pipe
        self._publisher_proc = await asyncio.create_subprocess_exec(
            *publisher_cmd,
            stdin=read_fd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        os.close(read_fd)  # Parent doesn't read; publisher owns it
        self._pipe_read_fd = None

        # Monitor publisher stderr for status messages
        self._stderr_task = asyncio.create_task(
            self._read_publisher_stderr(),
            name="overshoot-gopub-stderr",
        )

        # Monitor both processes
        self._on_ended = asyncio.get_event_loop().create_future()
        self._monitor_task = asyncio.create_task(
            self._monitor_processes(),
            name="overshoot-gopub-monitor",
        )

    async def _read_publisher_stderr(self) -> None:
        """Read JSON status messages from the Go publisher's stderr.

        The Go binary redirects pion/ICE debug noise to /dev/null internally,
        so only our JSON status messages should arrive here. As defense-in-depth,
        non-JSON lines are silently discarded and we yield to the event loop
        periodically to prevent starvation under unexpected load.
        """
        if not self._publisher_proc or not self._publisher_proc.stderr:
            return
        lines_since_yield = 0
        try:
            while True:
                line = await self._publisher_proc.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if not text or not text.startswith("{"):
                    # Fast-discard non-JSON lines (pion debug noise)
                    lines_since_yield += 1
                    if lines_since_yield >= 50:
                        await asyncio.sleep(0)
                        lines_since_yield = 0
                    continue
                try:
                    msg = json.loads(text)
                    event = msg.get("event", "")
                    if msg.get("error"):
                        logger.error("Go publisher [%s]: %s", event, msg["error"])
                    elif event == "connected":
                        logger.info("Go publisher connected: %s", msg.get("message", ""))
                    elif event == "publishing":
                        logger.info("Go publisher publishing: %s", msg.get("message", ""))
                    elif event == "eof":
                        logger.warning("Go publisher: input stream ended")
                    elif event == "shutdown":
                        logger.info("Go publisher shutting down")
                    else:
                        logger.debug("Go publisher [%s]: %s", event, msg.get("message", ""))
                except json.JSONDecodeError:
                    pass  # Silently discard malformed lines
                lines_since_yield += 1
                if lines_since_yield >= 50:
                    await asyncio.sleep(0)
                    lines_since_yield = 0
        except asyncio.CancelledError:
            pass

    async def _monitor_processes(self) -> None:
        """Wait for either process to exit and set the ended future."""
        assert self._ffmpeg_proc is not None
        assert self._publisher_proc is not None
        try:
            # Wait for either process to exit
            ffmpeg_task = asyncio.create_task(self._ffmpeg_proc.wait())
            publisher_task = asyncio.create_task(self._publisher_proc.wait())

            done, pending = await asyncio.wait(
                {ffmpeg_task, publisher_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

            # Determine which died and why
            if ffmpeg_task in done:
                rc = ffmpeg_task.result()
                if rc != 0:
                    error = SourceEndedError(
                        f"FFmpeg exited with code {rc} (source: {self.source})"
                    )
                else:
                    error = SourceEndedError(
                        f"FFmpeg ended (source: {self.source})"
                    )
            else:
                rc = publisher_task.result()
                error = SourceEndedError(
                    f"Go publisher exited with code {rc}"
                )

            if self._on_ended and not self._on_ended.done():
                self._on_ended.set_result(error)

        except asyncio.CancelledError:
            pass

    @property
    def ended_future(self) -> Optional[asyncio.Future[SourceEndedError]]:
        """Future that resolves when the pipeline ends unexpectedly."""
        return self._on_ended

    async def stop(self) -> None:
        """Stop both processes and clean up."""
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
            self._stderr_task = None

        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        for name, proc in [("publisher", self._publisher_proc), ("ffmpeg", self._ffmpeg_proc)]:
            if proc is not None:
                try:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except (ProcessLookupError, asyncio.TimeoutError):
                    proc.kill()
                logger.debug("Stopped %s process", name)

        self._ffmpeg_proc = None
        self._publisher_proc = None
