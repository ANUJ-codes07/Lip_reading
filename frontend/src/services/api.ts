const API_BASE_URL = 'http://localhost:8000/api';

export interface HealthResponse {
  status: string;
}

export interface RecognitionResponse {
  status: string;
  text: string;
  message?: string;
  inference_time?: string;
  file_size_bytes?: number;
}

/**
 * Checks connectivity with the FastAPI backend.
 */
export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      return false;
    }
    const data: HealthResponse = await response.json();
    return data.status === 'ok';
  } catch {
    return false;
  }
}

/**
 * Sends a captured video segment sequentially to the backend.
 * Recordings are transmitted as FormData and are not saved permanently.
 */
export async function sendVideoSegment(videoBlob: Blob): Promise<RecognitionResponse> {
  const formData = new FormData();
  // Determine appropriate filename extension from MIME type
  const extension = videoBlob.type.includes('mp4') ? 'mp4' : 'webm';
  formData.append('file', videoBlob, `segment.${extension}`);

  const response = await fetch(`${API_BASE_URL}/recognize`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let errorMessage = `Upload failed (${response.status})`;
    try {
      const errorJson = await response.json();
      if (errorJson && typeof errorJson.detail === 'string') {
        errorMessage = `Upload failed (${response.status}): ${errorJson.detail}`;
      } else {
        errorMessage = `Upload failed (${response.status}): ${JSON.stringify(errorJson)}`;
      }
    } catch {
      const text = await response.text();
      if (text) {
        errorMessage = `Upload failed (${response.status}): ${text}`;
      }
    }
    throw new Error(errorMessage);
  }

  return response.json();
}
