"""
Nano Banana Pro Image Generation Module
Premium image generation using Kie AI API (3 credits per image)
Supports higher resolution and better quality output

Copied from Algora (image_generation/nano_banana_pro.py). The hardcoded
fallback API key from the source has been REMOVED here — set KIE_API_KEY
in the environment instead. See README.md for setup.
"""

import os
import logging
import requests

logger = logging.getLogger(__name__)

# Configuration — KIE_API_KEY MUST be set in the environment.
KIE_API_KEY = os.environ.get('KIE_API_KEY')
KIE_API_BASE_URL = "https://api.kie.ai/api/v1/jobs"

# Model configuration
MODEL_NAME = "nano-banana-pro"
CREDIT_COST = 3  # 3 credits per generation

# Aspect ratio mapping
ASPECT_RATIO_MAP = {
    '1:1': '1:1',
    '16:9': '16:9',
    '9:16': '9:16',
    '4:3': '4:3',
    '3:4': '3:4',
    '3:2': '3:2',
    '2:3': '2:3',
    '5:4': '5:4',
    '4:5': '4:5',
    '21:9': '21:9',
    'auto': 'auto'
}


def _require_key():
    if not KIE_API_KEY:
        raise RuntimeError(
            "KIE_API_KEY environment variable is required. "
            "Get one from https://kie.ai and set KIE_API_KEY=sk-... in your env."
        )


def create_task(prompt, aspect_ratio="1:1", resolution="1K", output_format="png",
                image_input=None, callback_url=None):
    """
    Create a new image generation task with Nano Banana Pro model

    Args:
        prompt: Text description of the image (max 20000 characters)
        aspect_ratio: Aspect ratio (1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9, auto)
        resolution: Resolution (1K, 2K, 4K)
        output_format: Output format (png, jpg)
        image_input: Optional list of image URLs for reference (up to 8 images)
        callback_url: Optional callback URL for completion notification

    Returns:
        dict: Task creation result with taskId
    """
    _require_key()
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {KIE_API_KEY}"
        }

        # Map aspect ratio
        kie_aspect_ratio = ASPECT_RATIO_MAP.get(aspect_ratio, '1:1')

        payload = {
            "model": MODEL_NAME,
            "input": {
                "prompt": prompt[:20000],  # Max 20000 chars
                "aspect_ratio": kie_aspect_ratio,
                "resolution": resolution,
                "output_format": output_format
            }
        }

        if image_input and len(image_input) > 0:
            payload["input"]["image_input"] = image_input[:8]  # Max 8 images

        if callback_url:
            payload["callBackUrl"] = callback_url

        logger.info(f"Creating Nano Banana Pro task: {prompt[:50]}...")

        response = requests.post(
            f"{KIE_API_BASE_URL}/createTask",
            headers=headers,
            json=payload,
            timeout=30
        )

        result = response.json()

        if result.get("code") == 200:
            task_id = result.get("data", {}).get("taskId")
            logger.info(f"Nano Banana Pro task created: {task_id}")
            return {
                "success": True,
                "task_id": task_id,
                "model": MODEL_NAME,
                "credit_cost": CREDIT_COST
            }
        else:
            error_msg = result.get("message", "Unknown error")
            logger.error(f"Nano Banana Pro task creation failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }

    except requests.exceptions.RequestException as e:
        logger.error(f"Nano Banana Pro API request failed: {e}")
        return {
            "success": False,
            "error": f"API request failed: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Nano Banana Pro task creation error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def query_task(task_id):
    """
    Query task status and results

    Args:
        task_id: The task ID to query

    Returns:
        dict: Task status and results
    """
    _require_key()
    try:
        headers = {
            "Authorization": f"Bearer {KIE_API_KEY}"
        }

        response = requests.get(
            f"{KIE_API_BASE_URL}/recordInfo",
            headers=headers,
            params={"taskId": task_id},
            timeout=30
        )

        result = response.json()

        if result.get("code") == 200:
            data = result.get("data", {})
            state = data.get("state", "unknown")

            response_data = {
                "success": True,
                "task_id": task_id,
                "state": state,
                "model": data.get("model"),
                "cost_time": data.get("costTime"),
                "create_time": data.get("createTime"),
                "complete_time": data.get("completeTime")
            }

            if state == "success":
                # Parse resultJson to get image URLs
                result_json = data.get("resultJson")
                if result_json:
                    import json
                    try:
                        parsed = json.loads(result_json) if isinstance(result_json, str) else result_json
                        response_data["images"] = parsed.get("resultUrls", [])
                    except Exception:
                        response_data["images"] = []

            elif state == "fail":
                response_data["error"] = data.get("failMsg", "Generation failed")
                response_data["fail_code"] = data.get("failCode")

            return response_data
        else:
            return {
                "success": False,
                "error": result.get("message", "Query failed")
            }

    except Exception as e:
        logger.error(f"Nano Banana Pro query error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def get_credit_cost():
    """Return the credit cost for this model"""
    return CREDIT_COST


def get_model_name():
    """Return the model name"""
    return MODEL_NAME
