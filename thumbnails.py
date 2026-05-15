"""
Thumbnail generation via Nano Banana Pro (Kie API).

Thin convenience layer over nano_banana_pro: defaults tuned for YouTube-style
thumbnails (16:9, 2K), and a synchronous helper that polls until the job
finishes (or times out).
"""

import logging
import time

from nano_banana_pro import create_task, query_task

logger = logging.getLogger(__name__)

# YouTube thumbnail conventions
DEFAULT_ASPECT_RATIO = "16:9"
DEFAULT_RESOLUTION = "2K"        # 2K (2048×1152) is plenty for thumbnails;
                                 # 4K is overkill and ~4× the cost in time.
DEFAULT_OUTPUT_FORMAT = "png"
DEFAULT_POLL_INTERVAL_S = 3
DEFAULT_TIMEOUT_S = 300          # Nano Banana Pro usually returns in 30–90s.


def generate_thumbnail(
    prompt: str,
    *,
    reference_images: list[str] | None = None,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    resolution: str = DEFAULT_RESOLUTION,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    poll_interval_s: int = DEFAULT_POLL_INTERVAL_S,
) -> dict:
    """Submit a thumbnail prompt and block until the result is ready.

    Returns:
        {"success": True,  "task_id": ..., "images": [url, ...]}    on success
        {"success": False, "task_id": ..., "error":  "..."}         on failure
        {"success": False, "task_id": ..., "error":  "timeout"}     if it
                                                                    didn't
                                                                    finish in
                                                                    timeout_s

    `reference_images` accepts up to 8 publicly-fetchable image URLs that
    Nano Banana Pro will use as visual references (style, composition,
    character likeness etc). Skip for pure text-to-image.
    """
    submit = create_task(
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        output_format=output_format,
        image_input=reference_images or None,
    )
    if not submit.get("success"):
        return {"success": False, "task_id": None, "error": submit.get("error")}

    task_id = submit["task_id"]
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        status = query_task(task_id)
        if not status.get("success"):
            # transient query failure — wait and retry
            time.sleep(poll_interval_s)
            continue

        state = status.get("state")
        if state == "success":
            return {
                "success": True,
                "task_id": task_id,
                "images": status.get("images", []),
                "cost_time_s": (status.get("cost_time") or 0) / 1000,
            }
        if state == "fail":
            return {
                "success": False,
                "task_id": task_id,
                "error": status.get("error", "generation failed"),
                "fail_code": status.get("fail_code"),
            }
        # state in (waiting | running | unknown) — keep polling
        time.sleep(poll_interval_s)

    return {"success": False, "task_id": task_id, "error": "timeout"}


if __name__ == "__main__":
    # Manual smoke test — `python thumbnails.py "a cat in a spacesuit"`
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) < 2:
        print("usage: python thumbnails.py '<prompt>'")
        sys.exit(1)
    result = generate_thumbnail(sys.argv[1])
    print(result)
