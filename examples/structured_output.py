"""Example: Structured JSON output with an output schema.

Demonstrates using output_schema to get structured responses
that can be parsed as JSON.

Usage:
    python examples/structured_output.py --api-key YOUR_KEY --video /path/to/video.mp4
"""

import argparse
import asyncio
import json
import sys

import overshoot

SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {"type": "string"},
        },
        "description": {"type": "string"},
    },
    "required": ["objects", "description"],
}


async def main(api_key: str, video_path: str, model: str | None, duration: int) -> int:
    if model is None:
        models = await overshoot.get_models(api_key=api_key)
        ready = [m for m in models if m.ready]
        if not ready:
            print("No ready models available")
            return 1
        model = ready[0].model
        print(f"Auto-selected model: {model}")

    print(f"Video: {video_path}")
    print(f"Schema: {json.dumps(SCHEMA, indent=2)}\n")

    result_count = 0

    def on_result(r: overshoot.StreamInferenceResult) -> None:
        nonlocal result_count
        result_count += 1
        if r.ok:
            try:
                data = r.result_json()
                print(f"  Objects: {data.get('objects', [])}")
                print(f"  Description: {data.get('description', '')}")
                print()
            except (ValueError, json.JSONDecodeError):
                print(f"  Raw (not JSON): {r.result[:150]}\n")
        else:
            print(f"  ERROR: {r.error}\n")

    def on_error(e: Exception) -> None:
        print(f"  STREAM ERROR: {e}")

    client = overshoot.Overshoot(api_key=api_key)
    try:
        stream = await client.streams.create(
            source=overshoot.FileSource(path=video_path, loop=True),
            prompt="List the objects you see and describe the scene.",
            model=model,
            on_result=on_result,
            on_error=on_error,
            output_schema=SCHEMA,
            delay_seconds=2.0,
        )
        print(f"Stream started: {stream.stream_id}\n")

        await asyncio.sleep(duration)
        await stream.close()
    finally:
        await client.close()

    print(f"Received {result_count} results")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overshoot structured output example")
    parser.add_argument("--api-key", required=True, help="Overshoot API key")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--model", default=None, help="Model name (optional)")
    parser.add_argument("--duration", type=int, default=15, help="Duration in seconds")
    args = parser.parse_args()

    sys.exit(asyncio.run(main(args.api_key, args.video, args.model, args.duration)))
