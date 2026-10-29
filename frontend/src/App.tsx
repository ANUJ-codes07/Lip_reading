import React, { useEffect, useState } from 'react';
import { CameraView } from './components/CameraView';
import { TranscriptView } from './components/TranscriptView';
import { checkBackendHealth, RecognitionResponse, sendVideoSegment } from './services/api';

export const App: React.FC = () => {
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [lastResponse, setLastResponse] = useState<RecognitionResponse | null>(null);

  // Poll backend health on mount and periodically
  const verifyHealth = async () => {
    const isHealthy = await checkBackendHealth();
    setBackendOnline(isHealthy);
  };

  useEffect(() => {
    verifyHealth();
    const interval = setInterval(verifyHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  // Handle recorded video chunk sequentially
  const handleSegmentRecorded = async (blob: Blob) => {
    // Guard against duplicate or concurrent uploads
    if (isUploading) return;

    // Reset previous response and clear errors
    setLastResponse(null);
    setUploadError(null);

    // Enforce 50 MB limit on the client side before transmitting (accommodates ~1 min recordings)
    const MAX_CLIENT_BYTES = 50 * 1024 * 1024;
    if (blob.size > MAX_CLIENT_BYTES) {
      setUploadError(
        `Recorded segment (${(blob.size / (1024 * 1024)).toFixed(1)} MB) exceeds the 50 MB limit. Please record a shorter sample.`
      );
      return;
    }

    setIsUploading(true);

    try {
      const response = await sendVideoSegment(blob);
      setLastResponse(response);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown upload error occurred';
      setUploadError(message);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <div className="header-title">
          <h1>Visual Speech Recognition</h1>
          <p>Assisted communication through camera-based lip reading analysis</p>
        </div>

        <div className="status-badge">
          <span
            className={`status-dot ${
              backendOnline === null ? 'loading' : backendOnline ? 'online' : 'offline'
            }`}
          />
          <span>
            {backendOnline === null
              ? 'Connecting...'
              : backendOnline
              ? 'Backend Online'
              : 'Backend Offline'}
          </span>
        </div>
      </header>

      <main className="main-content">
        <CameraView
          onSegmentRecorded={handleSegmentRecorded}
          isUploading={isUploading}
        />

        <TranscriptView
          backendOnline={backendOnline}
          isUploading={isUploading}
          uploadError={uploadError}
          lastResponse={lastResponse}
        />
      </main>
    </div>
  );
};

export default App;
