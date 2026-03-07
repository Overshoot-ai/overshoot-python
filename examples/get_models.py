"""Example: List available models and their status.

Usage:
    python examples/get_models.py --api-key YOUR_KEY
"""

import argparse
import asyncio
import sys

import overshoot


async def main(api_key: str) -> int:
    models = await overshoot.get_models(api_key=api_key)

    print(f"Available models ({len(models)}):\n")
    for m in models:
        print(f"  {m.model:<50s} {m.status:<12s} {'READY' if m.ready else ''}")

    ready = [m for m in models if m.ready]
    print(f"\n{len(ready)} ready / {len(models)} total")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List Overshoot models")
    parser.add_argument("--api-key", required=True, help="Overshoot API key")
    args = parser.parse_args()

    sys.exit(asyncio.run(main(args.api_key)))
