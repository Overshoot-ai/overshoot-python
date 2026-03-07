# Examples

All examples require an Overshoot API key passed via `--api-key`.

## List available models

```bash
python examples/get_models.py --api-key YOUR_KEY
```

## Analyze a video file

```bash
python examples/file_source.py --api-key YOUR_KEY --video /path/to/video.mp4
```

## Analyze a live RTSP camera

```bash
python examples/rtsp_source.py --api-key YOUR_KEY --rtsp-url rtsp://user:pass@host/stream
```

## Push custom frames (OpenCV, numpy, etc.)

```bash
python examples/frame_source.py --api-key YOUR_KEY
```

## Structured JSON output

```bash
python examples/structured_output.py --api-key YOUR_KEY --video /path/to/video.mp4
```
