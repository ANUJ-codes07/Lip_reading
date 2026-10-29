# ==============================================================================
# CHAPLIN GOOGLE COLAB GPU PIPELINE (RetinaFace + FAN Visual Speech Recognition)
# Run this entire script in a single Google Colab cell with GPU enabled:
# Runtime -> Change runtime type -> T4 GPU
# ==============================================================================

# Step 1: Install dependencies and Chaplin's native RetinaFace & FAN packages
print("[1/5] Installing core dependencies (takes ~10s)...")
!pip install -q fastapi uvicorn httpx python-multipart hydra-core omegaconf sentencepiece

print("[1/5] Setting up RetinaFace & FAN tracker (shallow clone in ~2s)...")
import os
import sys
import shutil

# Fast shallow clone skipping LFS
if not os.path.exists("/content/face_detection"):
    os.system("GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 -q https://github.com/hhj1897/face_detection.git /content/face_detection")

if not os.path.exists("/content/face_alignment"):
    os.system("GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 -q https://github.com/hhj1897/face_alignment.git /content/face_alignment")

# Consolidate face_alignment into the unified ibug package inside face_detection
src_alignment = "/content/face_alignment/ibug/face_alignment"
dst_alignment = "/content/face_detection/ibug/face_alignment"
if os.path.exists(src_alignment):
    shutil.copytree(src_alignment, dst_alignment, dirs_exist_ok=True)

# Register /content/face_detection directly into sys.path
if "/content/face_detection" not in sys.path:
    sys.path.insert(0, "/content/face_detection")

# Pure Python robust checkpoint downloader (auto-detects and replaces corrupt Git LFS text pointers)
import urllib.request

def ensure_checkpoint_file(url: str, target_path: str, min_bytes: int):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    needs_download = False
    if not os.path.exists(target_path) or os.path.getsize(target_path) < min_bytes:
        needs_download = True
    else:
        try:
            with open(target_path, "rb") as f:
                header = f.read(16)
                if header.startswith(b"version ") or b"git-lfs" in header:
                    needs_download = True
        except Exception:
            needs_download = True

    if needs_download:
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
            except Exception:
                pass
        filename = os.path.basename(target_path)
        print(f" -> Downloading {filename} ({min_bytes // (1024*1024)}MB+)...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(target_path, "wb") as out:
            shutil.copyfileobj(resp, out)

    actual_size = os.path.getsize(target_path)
    with open(target_path, "rb") as f:
        header = f.read(16)
        if header.startswith(b"version "):
            raise RuntimeError(f"Corrupt checkpoint {target_path} (Git LFS text pointer)")
    print(f" -> ✓ {os.path.basename(target_path)} verified ({actual_size / (1024*1024):.1f} MB)")

resnet_pth = "/content/face_detection/ibug/face_detection/retina_face/weights/Resnet50_Final.pth"
fan_pth = "/content/face_detection/ibug/face_alignment/fan/weights/2dfan2.pth"

ensure_checkpoint_file(
    "https://media.githubusercontent.com/media/hhj1897/face_detection/master/ibug/face_detection/retina_face/weights/Resnet50_Final.pth",
    resnet_pth,
    50000000
)
ensure_checkpoint_file(
    "https://raw.githubusercontent.com/hhj1897/face_alignment/master/ibug/face_alignment/fan/weights/2dfan2.pth",
    fan_pth,
    20000000
)

# Step 2: Download cloudflared for public tunneling without account/token
print("\n[2/5] Setting up Cloudflare tunnel binary...")
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i -E cloudflared-linux-amd64.deb > /dev/null 2>&1

# Step 3: Clone Chaplin repo and download Auto-AVSR weights
import subprocess

if not os.path.exists("chaplin"):
    print("\n[3/5] Cloning Chaplin repository...")
    !git clone -q https://github.com/amanvirparhar/chaplin.git

%cd /content/chaplin

# Run setup to download weights from Hugging Face (Amanvir/LRS3_V_WER19.1)
print("\n[3/5] Downloading Auto-AVSR model weights from Hugging Face...")
!bash setup.sh

# Patch Chaplin's hardcoded cuda:0 in pipelines/pipeline.py to support dynamic CPU/GPU
pipeline_py = "/content/chaplin/pipelines/pipeline.py"
if os.path.exists(pipeline_py):
    with open(pipeline_py, "r") as f:
        p_code = f.read()
    p_code = p_code.replace('LandmarksDetector(device="cuda:0")', 'LandmarksDetector(device=device)')
    with open(pipeline_py, "w") as f:
        f.write(p_code)

# Step 4: Import models and initialize GPU/CPU pipelines
sys.path.append("/content/chaplin")

import torch

# Fix PyTorch 2.6 default weights_only=True breaking legacy model checkpoints (safe against cell re-runs)
import torch.serialization
if not hasattr(torch, "_original_load_fn") or getattr(torch._original_load_fn, "__name__", "") == "_compat_torch_load":
    torch._original_load_fn = torch.serialization.load

def _compat_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return torch._original_load_fn(*args, **kwargs)

torch.load = _compat_torch_load

import numpy as np
import torchvision
import torchvision.io
import cv2

# Fix modern torchvision (>=0.22) having removed torchvision.io.read_video
def _compat_read_video(filename, pts_unit="sec"):
    cap = cv2.VideoCapture(filename)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    if not frames:
        return torch.empty((0, 0, 0, 3), dtype=torch.uint8), None, {}
    tensor_frames = torch.from_numpy(np.array(frames))
    return tensor_frames, None, {}

torchvision.io.read_video = _compat_read_video

import base64
import re
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import threading
from transformers import pipeline

print(f"\n[4/5] Initializing GPU pipelines (CUDA: {torch.cuda.is_available()})...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Pre-flight checkpoint verification
print(" -> Validating RetinaFace & FAN checkpoints before loading...")
try:
    _ = torch.load(resnet_pth, map_location="cpu")
    print(" -> ✓ RetinaFace checkpoint pre-test passed")
except Exception as e:
    print(f" -> ⚠ Re-fetching RetinaFace checkpoint due to: {e}")
    if os.path.exists(resnet_pth):
        os.remove(resnet_pth)
    ensure_checkpoint_file(
        "https://media.githubusercontent.com/media/hhj1897/face_detection/master/ibug/face_detection/retina_face/weights/Resnet50_Final.pth",
        resnet_pth,
        50000000
    )

try:
    _ = torch.load(fan_pth, map_location="cpu")
    print(" -> ✓ FAN checkpoint pre-test passed")
except Exception as e:
    print(f" -> ⚠ Re-fetching FAN checkpoint due to: {e}")
    if os.path.exists(fan_pth):
        os.remove(fan_pth)
    ensure_checkpoint_file(
        "https://raw.githubusercontent.com/hhj1897/face_alignment/master/ibug/face_alignment/fan/weights/2dfan2.pth",
        fan_pth,
        20000000
    )

# 1. Initialize Auto-AVSR with native RetinaFace + FAN 68-point lip tracker
from pipelines.pipeline import InferencePipeline

print(f" -> Loading Chaplin Auto-AVSR (RetinaFace + FAN) into {'GPU' if device != 'cpu' else 'CPU'}...")
config_path = "./configs/LRS3_V_WER19.1.ini"
vsr_model = InferencePipeline(
    config_filename=config_path,
    detector="retinaface",
    face_track=True,
    device=device
)
vsr_model.eval()
print(f" -> ✓ Auto-AVSR loaded with native RetinaFace + FAN 68-point lip tracking (Device: {device})!")

# 2. Initialize Qwen LLM Corrector (0.5B: fast 490MB download, runs in <0.2s)
print(" -> Loading lightweight Qwen LLM for homophene correction...")
corrector_llm = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    max_new_tokens=40,
    do_sample=False,
)
print(" -> ✓ Qwen LLM loaded!")

def correct_lipread_text(raw_text: str) -> str:
    """
    Applies Chaplin's exact LLM prompt to fix phoneme/homophene errors in raw lip reading.
    """
    if not raw_text or not raw_text.strip():
        return ""

    raw_clean = raw_text.strip()
    system_prompt = (
        "You are an assistant that helps make corrections to the output of a lipreading model. "
        "The text you receive was transcribed using visual lip-reading (no audio), so it will likely have phonetic mistakes. "
        "The input is in ALL-CAPS. Your output should be capitalized normally and NOT in all-caps.\n\n"
        "If something seems unusual, assume it was mistranscribed. Infer the words actually spoken. "
        "Do not add extra conversational filler or explanations. "
        "Output ONLY the corrected sentence."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Raw lip-read text: {raw_clean}\nCorrected text:"}
    ]

    try:
        prompt_formatted = corrector_llm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = corrector_llm(prompt_formatted)
        generated = outputs[0]["generated_text"]

        # Extract only the assistant output
        if "<|im_start|>assistant\n" in generated:
            result = generated.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
        else:
            result = generated.replace(prompt_formatted, "").strip()

        # Clean any stray formatting
        result = result.strip('"').strip("'").strip()
        return result
    except Exception as e:
        print(f"LLM correction warning: {e}")
        return raw_clean.capitalize()


# Step 5: FastAPI Web Server
app = FastAPI(title="Chaplin Colab API")

class VideoRequest(BaseModel):
    video_base64: str
    filename: str = "segment.webm"

@app.get("/health")
def health():
    return {"status": "ok", "engine": "Chaplin (Auto-AVSR RetinaFace + Qwen LLM)"}

@app.post("/predict")
async def predict_lip_reading(req: VideoRequest):
    try:
        video_data = base64.b64decode(req.video_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 video encoding")

    ext = ".webm" if "webm" in req.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(video_data)
        tmp_path = tmp.name

    # Auto-AVSR strictly requires constant 25 fps with correct timestamp synchronization
    mp4_path = tmp_path.rsplit(".", 1)[0] + "_25fps.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_path, "-an", "-vf", "fps=25", "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    target_path = mp4_path if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0 else tmp_path

    try:
        # Run Auto-AVSR visual speech recognition with RetinaFace + FAN 68-point tracking
        with torch.no_grad():
            raw_result = vsr_model(target_path)

        raw_text = str(raw_result).strip() if raw_result else ""
        print(f"Raw Auto-AVSR output: '{raw_text}'")

        if not raw_text:
            return {
                "status": "success",
                "raw_text": "",
                "text": "",
                "message": "No speech detected in this video segment.",
            }

        # Run Chaplin LLM homophene correction
        corrected_text = correct_lipread_text(raw_text)
        print(f"Corrected output: '{corrected_text}'")

        return {
            "status": "success",
            "raw_text": raw_text,
            "text": corrected_text,
        }
    except AssertionError as ae:
        print(f"Face tracking note: {ae}")
        return {
            "status": "success",
            "raw_text": "",
            "text": "",
            "message": "Face was not clearly detected in the video. Please face the camera directly in good lighting.",
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"\n[Recognition Error Details]:\n{tb}", flush=True)
        return {
            "status": "success",
            "raw_text": "",
            "text": "",
            "message": f"Lip movement could not be extracted ({type(e).__name__}: {str(e)}). Please face the camera directly.",
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(mp4_path):
            os.remove(mp4_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Step 6: Clean previous processes and pick a free port
!pkill -f cloudflared > /dev/null 2>&1 || true

import socket
import time

def find_available_port(default=8000):
    for p in [default, 8080, 8888, 9000, 9090]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('127.0.0.1', p))
            s.close()
            return p
        except OSError:
            continue
    s = socket.socket()
    s.bind(('', 0))
    p = s.getsockname()[1]
    s.close()
    return p

free_port = find_available_port(8000)
print(f"\n[5/5] Starting FastAPI on port {free_port}...")

def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=free_port, log_level="warning")

server_thread = threading.Thread(target=run_fastapi, daemon=True)
server_thread.start()
time.sleep(2)

# Step 7: Launch Cloudflare Tunnel
print(f"Starting Cloudflare Tunnel for port {free_port}...")
cmd = f"cloudflared tunnel --url http://127.0.0.1:{free_port}"
process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

public_url = None
for line in iter(process.stdout.readline, ''):
    match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
    if match:
        public_url = match.group(0)
        break

if public_url:
    print("\n" + "=" * 70)
    print("🚀 CHAPLIN COLAB PIPELINE (RETINAFACE) IS LIVE!")
    print(f"API Endpoint: {public_url}/predict")
    print("=" * 70)
    print("\nNext step: Copy and paste this into your local backend/.env:")
    print(f"VSR_API_URL={public_url}/predict")
    print("RECOGNITION_MODE=colab")
    print("=" * 70)
    print("\n[ACTIVE] Listening for video segments from your web app...")
    print("Press the Stop (■) button in Colab whenever you want to shut down the server.\n")

    # Keep the Colab cell actively running so the tunnel never drops
    try:
        while True:
            line = process.stdout.readline()
            if line:
                if "Registered tunnel connection" in line or "error" in line.lower():
                    print(line.strip())
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Server stopped by user.")
else:
    print("Could not auto-detect tunnel URL. Check Cloudflare output logs above.")