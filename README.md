# thumbnails

AI thumbnail generation using **Nano Banana Pro** (Google's Gemini 2.5 Flash
Image) via the [Kie](https://kie.ai) API.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and put your KIE_API_KEY in
export $(cat .env | xargs)   # or use direnv / dotenv as you prefer
```

## Quick start

```bash
python thumbnails.py "Cinematic close-up of a hooded figure in fog, neon highlights, dramatic lighting"
```

…or from Python:

```python
from thumbnails import generate_thumbnail

result = generate_thumbnail(
    "Cinematic close-up of a hooded figure in fog, neon highlights",
    # Optional: style / character references (up to 8 image URLs)
    reference_images=["https://example.com/style-ref.png"],
)
print(result)
# {'success': True, 'task_id': '...', 'images': ['https://...png'], 'cost_time_s': 38.2}
```

## What's in here

| File | Role |
|---|---|
| `nano_banana_pro.py` | Thin wrapper over Kie's `/jobs/createTask` and `/jobs/recordInfo`. Mirrors the version in Algora's `image_generation/`. Async-style: returns a `task_id`, poll separately. |
| `thumbnails.py` | Convenience layer with thumbnail-friendly defaults (16:9, 2K, PNG) and a synchronous `generate_thumbnail` that polls until the task finishes. |
| `requirements.txt` | Just `requests`. |

## Defaults (override per call)

| Knob | Default | Notes |
|---|---|---|
| `aspect_ratio` | `16:9` | YouTube thumbnail standard. |
| `resolution` | `2K` | 2K is plenty (~2048×1152). 4K is ~4× the cost in time and rarely worth it for thumbs. |
| `output_format` | `png` | Lossless. Use `jpg` if you want smaller files. |
| `timeout_s` | `300` | Most jobs return in 30–90 s. |
| `poll_interval_s` | `3` | Cheap to poll; speeds up perceived latency. |

## Aspect ratios supported

`1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`, `auto`

## Pricing

3 Kie credits per generation regardless of resolution.
