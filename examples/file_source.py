"""Example: Analyze a local video file with Overshoot.

Usage:
    python examples/file_source.py --api-key YOUR_KEY --video /path/to/video.mp4

Options:
    --api-key       Overshoot API key (required)
    --video         Path to a video file (required)
    --model         Model name (default: auto-selects first ready model)
    --duration      How long to run in seconds (default: 15)
"""

import argparse
import asyncio
import sys

import overshoot


async def main(api_key: str, video_path: str, model: str | None, duration: int) -> int:
    # If no model specified, pick the first ready one
    if model is None:
        models = await overshoot.get_models(api_key=api_key)
        ready = [m for m in models if m.ready]
        if not ready:
            print("No ready models available")
            return 1
        model = ready[0].model
        print(f"Auto-selected model: {model}")

    print(f"Video: {video_path}")
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
            source=overshoot.FileSource(path=video_path, loop=True),
            prompt="Describe what you see in this video briefly.",
            model=model,
            on_result=on_result,
            on_error=on_error,
        )
        print(f"Stream started: {stream.stream_id}\n")

        await asyncio.sleep(duration)
        await stream.close()
    finally:
        await client.close()

    print(f"\nReceived {result_count} results")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overshoot FileSource example")
    parser.add_argument("--api-key", required=True, help="Overshoot API key")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--model", default=None, help="Model name (optional)")
    parser.add_argument("--duration", type=int, default=15, help="Duration in seconds")
    args = parser.parse_args()

    sys.exit(asyncio.run(main(args.api_key, args.video, args.model, args.duration)))
