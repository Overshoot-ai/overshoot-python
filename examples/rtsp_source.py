"""Example: Analyze a live RTSP camera stream with Overshoot.

Usage:
    python examples/rtsp_source.py --api-key YOUR_KEY --rtsp-url rtsp://user:pass@host/stream

Options:
    --api-key       Overshoot API key (required)
    --rtsp-url      RTSP stream URL (required)
    --model         Model name (default: auto-selects first ready model)
    --duration      How long to run in seconds (default: 15)
"""

import argparse
import asyncio
import sys

import overshoot


async def main(api_key: str, rtsp_url: str, model: str | None, duration: int) -> int:
    if model is None:
        models = await overshoot.get_models(api_key=api_key)
        ready = [m for m in models if m.ready]
        if not ready:
            print("No ready models available")
            return 1
        model = ready[0].model
        print(f"Auto-selected model: {model}")

    print(f"RTSP: {rtsp_url}")
    print(f"Duration: {duration}s\n")

    result_count = 0

    def on_result(r: overshoot.StreamInferenceResult) -> None:
        nonlocal result_count
        result_count += 1
        if r.ok:
            print(f"  [{r.total_latency_ms:.0f}ms] {r.result[:150]}")
        else:
            print(f"  ERROR: {r.error}")

    def on_error(e: Exception) -> None:
        print(f"  STREAM ERROR: {e}")

    client = overshoot.Overshoot(api_key=api_key)
    try:
        stream = await client.streams.create(
            source=overshoot.RTSPSource(url=rtsp_url),
            prompt="Describe what you see briefly.",
            model=model,
            on_result=on_result,
            on_error=on_error,
            target_fps=6,
            clip_length_seconds=1.0,
            delay_seconds=1.0,
        )
        print(f"Stream started: {stream.stream_id}\n")

        await asyncio.sleep(duration)
        await stream.close()
    finally:
        await client.close()

    print(f"\nReceived {result_count} results")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overshoot RTSPSource example")
    parser.add_argument("--api-key", required=True, help="Overshoot API key")
    parser.add_argument("--rtsp-url", required=True, help="RTSP stream URL")
    parser.add_argument("--model", default=None, help="Model name (optional)")
    parser.add_argument("--duration", type=int, default=15, help="Duration in seconds")
    args = parser.parse_args()

    sys.exit(asyncio.run(main(args.api_key, args.rtsp_url, args.model, args.duration)))
