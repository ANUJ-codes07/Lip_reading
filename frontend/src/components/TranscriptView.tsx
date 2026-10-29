import React from 'react';
import { RecognitionResponse } from '../services/api';

interface TranscriptViewProps {
  backendOnline: boolean | null;
  isUploading: boolean;
  uploadError: string | null;
  lastResponse: RecognitionResponse | null;
}

export const TranscriptView: React.FC<TranscriptViewProps> = ({
  backendOnline,
  isUploading,
  uploadError,
  lastResponse,
}) => {
  return (
    <div className="card">
      <h2 className="card-title">Recognized Speech Transcript</h2>

      <div className="transcript-area">
        {/* Backend health status */}
        <div className="info-item">
          <span className="info-label">Backend Connection:</span>
          <span className="info-value">
            {backendOnline === null
              ? 'Checking...'
              : backendOnline
              ? 'Connected (http://localhost:8000)'
              : 'Disconnected (ensure backend is running)'}
          </span>
        </div>

        {/* Current upload/recognition status */}
        <div className="info-item">
          <span className="info-label">Recognition State:</span>
          <span className="info-value">
            {isUploading
              ? 'Transcribing silent video with visual lip reader...'
              : lastResponse
              ? 'Recognition Complete'
              : 'Idle (Ready for recording)'}
          </span>
        </div>

        {/* Prominent error notification if request or external service failed */}
        {uploadError && (
          <div
            style={{
              backgroundColor: 'rgba(248, 113, 113, 0.12)',
              border: '1px solid var(--accent-red)',
              borderRadius: '8px',
              padding: '12px 14px',
              color: 'var(--text-primary)',
              fontSize: '0.9rem',
              lineHeight: 1.4,
            }}
          >
            <strong style={{ color: 'var(--accent-red)' }}>Recognition Failed: </strong>
            {uploadError}
          </div>
        )}

        {/* Prominent accessible caption transcript box */}
        {lastResponse && (
          <div
            style={{
              marginTop: '6px',
              backgroundColor: 'var(--bg-surface)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              padding: '16px',
              display: 'flex',
              flexDirection: 'column',
              gap: '10px',
            }}
          >
            <span
              style={{
                fontSize: '0.8rem',
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                color: 'var(--text-secondary)',
                fontWeight: 600,
              }}
            >
              Transcribed Speech Output:
            </span>

            {lastResponse.text ? (
              <div
                style={{
                  fontSize: '1.4rem',
                  fontWeight: 700,
                  color: 'var(--accent-blue)',
                  lineHeight: 1.3,
                  letterSpacing: '0.01em',
                  padding: '4px 0',
                }}
              >
                "{lastResponse.text}"
              </div>
            ) : (
              <div
                style={{
                  fontSize: '1rem',
                  color: 'var(--text-secondary)',
                  fontStyle: 'italic',
                }}
              >
                No intelligible speech detected in this segment.
              </div>
            )}

            {lastResponse.message && (
              <p
                className="status-message"
                style={{
                  borderTop: '1px solid var(--border-subtle)',
                  paddingTop: '8px',
                  fontSize: '0.8rem',
                }}
              >
                Status: {lastResponse.message}
              </p>
            )}
          </div>
        )}

        {/* Default idle message before any sample has been recorded */}
        {!lastResponse && !isUploading && !uploadError && (
          <p className="status-message" style={{ marginTop: '16px' }}>
            No speech segments processed yet. Start the camera and record a 2–3 second sample of spoken words to transcribe.
          </p>
        )}
      </div>
    </div>
  );
};
