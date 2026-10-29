import os
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from schemas import HealthResponse, RecognitionResponse
from services.recognition_service import process_video_segment

# Maximum allowed video upload size: 50 MB (accommodates recordings up to ~1 minute)
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

# Allowed frontend origin during development
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "http://localhost:5173")

app = FastAPI(
    title="Lip Reading Backend",
    description="Backend API for visual speech recognition segment handling",
    version="0.1.0",
)

# Configure CORS restricted only to the local frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify frontend-to-backend connectivity.
    """
    return HealthResponse(status="ok")


@app.post("/api/recognize", response_model=RecognitionResponse)
async def recognize_video_segment(file: UploadFile = File(...)):
    """
    Receives a sequential video segment from the frontend.
    Validates file size and delegates processing to the recognition service.
    Recordings are kept in memory and are NOT saved to disk.
    """
    # Validate content type starts with video/
    content_type = file.content_type or "video/webm"
    if not content_type.startswith("video/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type '{content_type}'. Please upload a video file.",
        )

    # Fast check if Content-Length / file size header is already present
    if file.size and file.size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Video segment exceeds maximum permitted size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    # Read up to MAX_FILE_SIZE_BYTES + 1 to prevent memory exhaustion from oversized payloads
    contents = await file.read(MAX_FILE_SIZE_BYTES + 1)
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Video segment exceeds maximum permitted size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded video segment is empty.",
        )

    # Forward to visual speech recognition service
    response = await process_video_segment(video_bytes=contents, content_type=content_type)
    return response
