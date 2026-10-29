from typing import Optional
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Schema for server health status check."""
    status: str


class RecognitionResponse(BaseModel):
    """
    Schema for visual speech recognition responses.
    Contains actual recognized text returned by the model.
    """
    status: str
    text: str
    message: Optional[str] = None
    inference_time: Optional[str] = None
    file_size_bytes: Optional[int] = None
