import base64
import os
import httpx
from fastapi import HTTPException, status

from schemas import RecognitionResponse


def _read_env_setting(key: str, default: str = "") -> str:
    """
    Retrieves setting from backend/.env directly first so updates take immediate effect,
    falling back to os.getenv.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(current_dir)
    env_file = os.path.join(backend_dir, ".env")

    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"{key}="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
        except Exception:
            pass

    val = os.getenv(key)
    if val:
        return val.strip()

    return default


async def _process_with_auto_avsr(video_bytes: bytes, content_type: str, vsr_api_url: str, vsr_api_key: str = "") -> RecognitionResponse:
    """
    Relays the video segment to the Chaplin Auto-AVSR inference endpoint on Google Colab.
    """
    b64_video = base64.b64encode(video_bytes).decode("utf-8")
    ext = "mp4" if "mp4" in content_type else "webm"

    payload = {
        "video_base64": b64_video,
        "filename": f"segment.{ext}",
    }

    headers = {"Content-Type": "application/json"}
    if vsr_api_key:
        headers["Authorization"] = f"Bearer {vsr_api_key}"

    timeout = httpx.Timeout(300.0, connect=15.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(vsr_api_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Unable to connect to Chaplin Auto-AVSR service at '{vsr_api_url}'. Please ensure your Google Colab tunnel is running.",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Chaplin Auto-AVSR service timed out while processing video segment.",
        )
    except httpx.HTTPStatusError as err:
        err_msg = err.response.text
        try:
            err_json = err.response.json()
            err_msg = err_json.get("detail") or err_json.get("message") or err_msg
        except Exception:
            pass
        # Clean any multi-line stack trace down to a readable error summary
        if "Traceback" in err_msg:
            err_msg = err_msg.splitlines()[-1]
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Chaplin Auto-AVSR error ({err.response.status_code}): {err_msg}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error communicating with Chaplin service: {str(e)}",
        )

    raw_text = (data.get("raw_text") or "").strip()
    recognized_text = (data.get("text") or "").strip()
    message = (data.get("message") or "").strip()

    if raw_text and recognized_text:
        status_detail = f"Chaplin Auto-AVSR: '{raw_text}' -> '{recognized_text}'"
    elif recognized_text:
        status_detail = f"Chaplin Auto-AVSR: '{recognized_text}'"
    elif message:
        status_detail = message
    else:
        status_detail = "No intelligible speech detected in this video segment."

    return RecognitionResponse(
        status="success",
        text=recognized_text,
        message=status_detail,
        file_size_bytes=len(video_bytes),
    )


async def process_video_segment(video_bytes: bytes, content_type: str) -> RecognitionResponse:
    """
    Directly dispatches video segments to the Chaplin Auto-AVSR pipeline.
    """
    vsr_api_url = _read_env_setting("VSR_API_URL")
    vsr_api_key = _read_env_setting("VSR_API_KEY")

    if not vsr_api_url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="VSR_API_URL is not set. Please set VSR_API_URL in backend/.env to your running Colab Cloudflare endpoint.",
        )

    return await _process_with_auto_avsr(video_bytes, content_type, vsr_api_url, vsr_api_key)
