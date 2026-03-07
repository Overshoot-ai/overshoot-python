"""Example: Push custom frames to Overshoot for analysis.

Demonstrates FrameSource with programmatic frame injection.
This example generates solid-color frames. In practice you would
push frames from OpenCV, PIL, a robotics stack, etc.

Usage:
    python examples/frame_source.py --api-key YOUR_KEY

Options:
    --api-key       Overshoot API key (required)
    --model         Model name (default: auto-selects first ready model)
    --duration      How long to run in seconds (default: 15)
"""

import argparse
import asyncio
import sys

import overshoot


async def main(api_key: str, model: str | None, duration: int) -> int:
    if model is None:
        models = await overshoot.get_models(api_key=api_key)
        ready = [m for m in models if m.ready]
        if not ready:
            print("No ready models available")
            return 1
        model = ready[0].model
        print(f"Auto-selected model: {model}")

    print(f"Duration: {duration}s\n")

    width, height = 640, 480
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

    source = overshoot.FrameSource(width=width, height=height)
    client = overshoot.Overshoot(api_key=api_key)

    try:
        stream = await client.streams.create(
            source=source,
            prompt="Describe what you see. If it's a solid color, say which color.",
            model=model,
            on_result=on_result,
            on_error=on_error,
            delay_seconds=1.0,
        )
        print(f"Stream started: {stream.stream_id}")
        print(f"Pushing {width}x{height} frames at ~10fps...\n")

        # Push alternating red / blue frames at ~10fps
        frame_count = 0
        end_time = asyncio.get_event_loop().time() + duration
        while asyncio.get_event_loop().time() < end_time and stream.is_active:
            if frame_count % 20 < 10:
                pixel = b"\xff\x00\x00\xff"  # Red RGBA
            else:
                pixel = b"\x00\x00\xff\xff"  # Blue RGBA

            source.push_frame(pixel * (width * height))
            frame_count += 1
            await asyncio.sleep(0.1)

        print(f"\nPushed {frame_count} frames")
        await stream.close()
    finally:
        await client.close()

    print(f"Received {result_count} results")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overshoot FrameSource example")
    parser.add_argument("--api-key", required=True, help="Overshoot API key")
    parser.add_argument("--model", default=None, help="Model name (optional)")
    parser.add_argument("--duration", type=int, default=15, help="Duration in seconds")
    args = parser.parse_args()

    sys.exit(asyncio.run(main(args.api_key, args.model, args.duration)))
