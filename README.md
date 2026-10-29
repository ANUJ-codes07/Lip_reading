# 🗣️ Visual Speech Recognition — AI Lip Reading System

> *Translating silent lip movements into words using deep learning, without relying on audio.*

---

## 📖 Table of Contents

- [Overview](#overview)
- [How It Works — The Full Pipeline](#how-it-works--the-full-pipeline)
- [Neural Network Architecture](#neural-network-architecture)
- [Inference Modes](#inference-modes)
  - [Local Mode (RTX GPU)](#-local-mode-rtx-gpu)
  - [Colab Mode (Cloud GPU)](#-colab-mode-cloud-gpu)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Technology Stack](#technology-stack)

---

## Overview

This project is an end-to-end **Visual Speech Recognition (VSR)** system — commonly known as **lip reading** — that transcribes spoken words purely from video input of a speaker''s face. No microphone. No audio. Just visual information.

It is built on top of [Auto-AVSR](https://github.com/mpc001/auto_avsr), a state-of-the-art Audio-Visual Speech Recognition framework trained on the LRS3 benchmark dataset. The system achieves a **Word Error Rate (WER) of ~19.1%** on LRS3 — one of the best publicly available open-source lip reading models.

The application consists of three layers:

| Layer | Technology | Responsibility |
|---|---|---|
| **Frontend** | React + TypeScript + Vite | Live webcam capture, video recording, and transcript display |
| **Backend (Orchestrator)** | FastAPI (Python) | Video ingestion, validation, and routing to the inference engine |
| **Inference Engine** | PyTorch + Auto-AVSR | Face detection, landmark extraction, mouth cropping, and neural decoding |

---

## How It Works — The Full Pipeline

When you record a lip-reading segment in the browser, the system processes it through a rigorous 6-step pipeline:

```
[Browser Webcam] ──► [Video Segment (WebM)] ──► [FastAPI Backend]
                                                       │
                                          ┌────────────▼────────────┐
                                          │   Step 1: FFmpeg         │
                                          │   Normalize to 25.0 FPS  │
                                          │   H.264 MP4, reset PTS   │
                                          └────────────┬────────────┘
                                                       │
                                          ┌────────────▼────────────┐
                                          │   Step 2: Frame Loading  │
                                          │   OpenCV BGR → RGB       │
                                          └────────────┬────────────┘
                                                       │
                                          ┌────────────▼────────────┐
                                          │   Step 3: Face Tracking  │◄── Neural Network
                                          │   RetinaFace (ResNet-50) │
                                          │   FAN 68-Point Landmarks │
                                          └────────────┬────────────┘
                                                       │
                                          ┌────────────▼────────────┐
                                          │   Step 4: V-VAD          │
                                          │   Mouth Aspect Ratio     │
                                          │   Velocity-based VAD     │
                                          └────────────┬────────────┘
                                                       │
                                          ┌────────────▼────────────┐
                                          │   Step 5: Mouth Crop     │
                                          │   Affine-aligned 96x96px │
                                          │   CenterCrop 88x88px     │
                                          └────────────┬────────────┘
                                                       │
                                          ┌────────────▼────────────┐
                                          │   Step 6: VSR Inference  │◄── Neural Network
                                          │   Conformer Encoder       │
                                          │   CTC + Beam Search (40) │
                                          │   LM Shallow Fusion      │
                                          └────────────┬────────────┘
                                                       │
                                          ┌────────────▼────────────┐
                                          │   Post-Processing        │
                                          │   Homophene Correction   │
                                          │   LLM Rescoring (Colab)  │
                                          └────────────┬────────────┘
                                                       │
                                               [Transcribed Text]
                                                       │
                                          ┌────────────▼────────────┐
                                          │   React Frontend         │
                                          │   Live Subtitle Display  │
                                          └─────────────────────────┘
```

---

## Neural Network Architecture

This is where the advanced deep learning is at the core of the system. Multiple neural networks work in sequence:

### 1. RetinaFace — Face Detection (ResNet-50 Backbone)

**Used in:** Both Local and Colab modes.

RetinaFace is a state-of-the-art single-stage face detector based on a **ResNet-50** convolutional backbone. It operates frame-by-frame on the video to localize faces with sub-pixel precision. It outputs bounding boxes and five facial anchor points used to initialize landmark tracking.

- Architecture: Feature Pyramid Network (FPN) with multi-task learning
- Training: WIDER FACE dataset
- Role: Robustly finds the face even under varying lighting, angle, and occlusion

### 2. FAN — 68-Point Facial Landmark Detection (Face Alignment Network)

**Used in:** Both Local and Colab modes.

After RetinaFace locates the face, the **Face Alignment Network (FAN)** predicts **68 precise facial landmark coordinates** — specifically around the lip region (points 48–68). These landmarks are the foundation for:

- Calculating the **Mouth Aspect Ratio (MAR)** for visual activity detection
- Performing **affine alignment** to normalize the mouth region across frames
- Cropping a stable **96×96 pixel mouth patch** invariant to head movement

The landmark coordinates also power the **Visual Voice Activity Detection (V-VAD)** system — a custom velocity-based algorithm that detects whether the person is actually speaking or silent, preventing the model from running inference on empty frames.

### 3. Auto-AVSR Conformer — The Core Lip Reading Model

**Used in:** Both Local and Colab modes.

This is the primary neural network responsible for decoding lip movements into text. It is a **Conformer-based sequence-to-sequence model** — a hybrid architecture combining:

- **Convolutional Neural Network (CNN) Frontend:** Extracts spatial lip-motion features from the 88×88 grayscale mouth patch tensor at each frame
- **Conformer Encoder:** Processes the temporal sequence of frame features. The Conformer (Convolution-augmented Transformer) architecture is superior to plain Transformers for speech tasks as it captures both local and global temporal dependencies simultaneously
- **CTC (Connectionist Temporal Classification) Decoder:** Enables alignment-free sequence prediction from the encoder output
- **Beam Search Decoder (beam size = 40):** Generates the top-N most likely transcription hypotheses from the model output, matching the LRS3 benchmark configuration

**Checkpoint:** `vsr_trlrs3_base.pth` — trained on LRS3 (Lip Reading in the Wild), achieving **~19.1% WER**

### 4. Language Model (LM) Shallow Fusion & Homophene Disambiguation

**Used in:** Local mode (statistical LM), Colab mode (neural LM via Qwen).

A critical post-processing stage that resolves a fundamental challenge in lip reading: **visual homophones** (words that look identical on the lips, such as /p/, /b/, /m/ or /t/, /d/, /n/).

**In Local Mode — Statistical LM Rescorer:**
- A custom n-gram language model rescorer re-ranks the beam search candidates
- Scoring formula: `Final Score = Visual_Score + (0.35 × LM_Score)`
- Uses a curated bigram/trigram collocation table of high-frequency conversational English
- Maintains a homophene confusion matrix to generate and evaluate visually plausible alternatives

**In Colab Mode — Neural LM via Qwen 2.5 (0.5B):**
- A quantized **Qwen 2.5 (0.5B-Instruct)** large language model provides context-aware phoneme correction
- Given the raw ALL-CAPS lip-read output, the LLM infers the most contextually plausible corrected sentence
- Runs in `float16` on GPU for near-instant correction (<0.2 seconds per segment)

---

## Inference Modes

The system supports two deployment modes for the AI inference engine, designed to accommodate both GPU-equipped local machines and GPU-free setups via cloud compute.

---

### Local Mode (RTX GPU)

**File:** `backend/local_chaplin_server.py`

**Best for:** Developers with an NVIDIA GPU (e.g., RTX 3050, 4060, 5050 or better) who want the lowest possible latency and full offline inference.

In this mode, the entire inference pipeline runs on your local machine using your GPU via CUDA. The server starts at `http://127.0.0.1:8001`.

**What happens:**
1. The server loads the `vsr_trlrs3_base.pth` checkpoint directly from disk
2. The RetinaFace and FAN models are loaded from the local `D:\lipsinc` virtual environment
3. All inference — face detection, landmark tracking, mouth cropping, conformer encoding, and beam search — runs on the local RTX GPU
4. Post-processing uses the **Statistical LM Rescorer** (no LLM or internet connection required)
5. CUDA memory is flushed after each prediction (`torch.cuda.empty_cache()`)

**How to run:**
```powershell
cd D:\Anuj31\backend
D:\lipsinc\.venv\Scripts\python.exe local_chaplin_server.py
```

The server will print the GPU name, checkpoint path, and FFmpeg binary path on startup.

---

### Colab Mode (Cloud GPU)

**File:** `backend/colab_chaplin_server.py`

**Best for:** Users without a powerful local GPU, or when running on a laptop. This mode offloads all heavy inference to a free Google Colab T4 GPU instance.

In this mode, `colab_chaplin_server.py` is pasted and run as a single cell in Google Colab. It:

1. **Installs all dependencies** (FastAPI, uvicorn, hydra, sentencepiece) in the Colab environment
2. **Clones RetinaFace and FAN** from GitHub (shallow clone, ~2 seconds)
3. **Downloads the Auto-AVSR model weights** from Hugging Face (`Amanvir/LRS3_V_WER19.1`)
4. **Downloads and installs Cloudflare Tunnel** (`cloudflared`) to expose the Colab server publicly via a `*.trycloudflare.com` HTTPS URL — no Ngrok account or token required
5. Starts the FastAPI inference server on port `8001` inside Colab
6. Outputs the public tunnel URL, which is then set in the local backend''s `.env` as `CHAPLIN_URL`
7. Additionally loads **Qwen 2.5 (0.5B-Instruct)** as a neural LM corrector for superior phoneme disambiguation

**How to run:**
1. Open [Google Colab](https://colab.research.google.com) and create a new notebook
2. Set Runtime → Change runtime type → **T4 GPU**
3. Paste the entire contents of `colab_chaplin_server.py` into a cell and run it
4. Copy the printed `*.trycloudflare.com` tunnel URL from the output
5. Set it in `backend/.env`:
   ```
   CHAPLIN_URL=https://your-tunnel-url.trycloudflare.com
   ```

---

## Project Structure

```
Anuj31/
├── backend/
│   ├── main.py                    # FastAPI orchestrator (video ingestion & routing)
│   ├── local_chaplin_server.py    # Local GPU inference engine (Auto-AVSR + Statistical LM)
│   ├── colab_chaplin_server.py    # Google Colab GPU inference engine (Auto-AVSR + Qwen LLM)
│   ├── schemas.py                 # Pydantic request/response schemas
│   ├── services/
│   │   └── recognition_service.py # Bridges main API to the active inference engine
│   ├── requirements.txt           # Python dependencies
│   └── .env                       # Environment configuration (CHAPLIN_URL, ALLOWED_ORIGIN)
│
└── frontend/
    ├── src/
    │   ├── App.tsx                # Root component — health polling & segment orchestration
    │   ├── components/
    │   │   ├── CameraView.tsx     # Webcam capture & video recording
    │   │   └── TranscriptView.tsx # Live transcript display & status
    │   ├── services/
    │   │   └── api.ts             # Backend API client (health check & video upload)
    │   └── index.css              # Global styles
    ├── package.json
    └── vite.config.ts
```

---

## Getting Started

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Node.js | 18+ |
| FFmpeg | Any recent version |
| CUDA (for Local mode) | 11.8+ |
| PyTorch | 2.0+ |

---

### Backend Setup

```powershell
# 1. Navigate to the backend directory
cd D:\Anuj31\backend

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and configure environment variables
copy .env.example .env
# Edit .env and set CHAPLIN_URL to your inference engine address

# 5. Start the FastAPI backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Then, in a separate terminal, start the inference engine (see [Inference Modes](#inference-modes)).

---

### Frontend Setup

```powershell
cd D:\Anuj31\frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:5173`.

---

## Configuration

The backend is configured via `backend/.env`:

```env
# URL of the active Chaplin inference engine
# For Local mode: http://127.0.0.1:8001
# For Colab mode: https://your-tunnel.trycloudflare.com
CHAPLIN_URL=http://127.0.0.1:8001

# Allowed CORS origin for the frontend
ALLOWED_ORIGIN=http://localhost:5173
```

---

## API Reference

### `GET /api/health`

Checks whether the backend server is reachable.

**Response:**
```json
{ "status": "ok" }
```

---

### `POST /api/recognize`

Accepts a video segment (WebM/MP4) and returns the transcribed text.

**Request:** `multipart/form-data` with a `file` field (video blob, max 50 MB)

**Response (speech detected):**
```json
{
  "status": "success",
  "text": "How are you",
  "raw_text": "HOW ARE YOU",
  "message": ""
}
```

**Response (no speech detected):**
```json
{
  "status": "success",
  "text": "",
  "message": "No speech detected (lips were still/silent)."
}
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React 18, TypeScript, Vite |
| Backend API | FastAPI, Uvicorn, Pydantic |
| Face Detection | RetinaFace (ResNet-50 + FPN) |
| Landmark Tracking | FAN — Face Alignment Network (68-point) |
| Core VSR Model | Auto-AVSR Conformer (LRS3, WER 19.1%) |
| Beam Search | ESPnet-style, beam size 40 |
| LM Correction (Local) | Custom Statistical N-gram Rescorer |
| LM Correction (Colab) | Qwen 2.5 — 0.5B-Instruct (float16) |
| Tunneling (Colab) | Cloudflare Tunnel (cloudflared) |
| Video Preprocessing | FFmpeg, OpenCV |
| GPU Runtime | PyTorch + CUDA |

---

## Acknowledgements

- [Auto-AVSR](https://github.com/mpc001/auto_avsr) — The foundational visual speech recognition framework
- [Chaplin](https://github.com/amanvirparhar/chaplin) — Auto-AVSR integration and model weights hosting
- [iBUG Face Detection & Alignment](https://github.com/hhj1897/face_detection) — RetinaFace and FAN implementations
- [Qwen 2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) — Lightweight LLM for phoneme correction
- LRS3 Dataset — Lip Reading in the Wild benchmark

---

*Built with the goal of making communication technology accessible to everyone, including those who cannot rely on audio.*
