import React, { useEffect, useRef, useState } from 'react';

interface CameraViewProps {
  onSegmentRecorded: (blob: Blob) => void;
  isUploading: boolean;
}

// Find a supported video container and codec for the current browser
const getPreferredMimeType = (): string => {
  const candidateTypes = [
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
    'video/mp4;codecs=avc1',
    'video/mp4',
  ];
  for (const type of candidateTypes) {
    if (MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }
  return ''; // Browser default
};

export const CameraView: React.FC<CameraViewProps> = ({ onSegmentRecorded, isUploading }) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);

  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isFinalizing, setIsFinalizing] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // Synchronize media stream with the video element whenever camera active state changes
  useEffect(() => {
    if (isCameraActive && videoRef.current && mediaStreamRef.current) {
      videoRef.current.srcObject = mediaStreamRef.current;
      videoRef.current.play().catch((err) => {
        console.warn('Video preview playback warning:', err);
      });
    } else if (!isCameraActive && videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, [isCameraActive]);

  const stopCamera = () => {
    // If currently recording, cancel cleanly without dispatching segment
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.ondataavailable = null;
      mediaRecorderRef.current.onstop = null;
      try {
        mediaRecorderRef.current.stop();
      } catch {
        // Ignore any errors during immediate stop
      }
    }
    mediaRecorderRef.current = null;
    recordedChunksRef.current = [];
    setIsRecording(false);
    setIsFinalizing(false);

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
  };

  // Stop camera tracks on unmount using ref values directly
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const startCamera = async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 25, max: 30 },
          facingMode: 'user',
        },
        audio: false, // Visual speech recognition focuses solely on lips/visuals
      });

      mediaStreamRef.current = stream;
      setIsCameraActive(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unable to access camera';
      setCameraError(`Camera permission denied or camera unavailable: ${message}`);
      setIsCameraActive(false);
    }
  };

  const startRecording = () => {
    if (!mediaStreamRef.current) return;

    setCameraError(null);
    recordedChunksRef.current = [];

    const preferredMime = getPreferredMimeType();
    const options: MediaRecorderOptions = {
      ...(preferredMime ? { mimeType: preferredMime } : {}),
      videoBitsPerSecond: 1_200_000, // 1.2 Mbps keeps 640x480 video crisp while under 1 MB for fast cloud ingestion
    };

    try {
      const recorder = new MediaRecorder(mediaStreamRef.current, options);

      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data && event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        setIsFinalizing(false);

        if (recordedChunksRef.current.length > 0) {
          const finalMime = recorder.mimeType || preferredMime || 'video/webm';
          const completeBlob = new Blob(recordedChunksRef.current, { type: finalMime });

          // Reject empty or sub-frame segments that cannot be processed
          if (completeBlob.size < 100) {
            setCameraError('Recording was too short to capture video data. Please record for at least 1–2 seconds.');
            return;
          }

          onSegmentRecorded(completeBlob);
        } else {
          setCameraError('Recording was too short to capture video data. Please record for at least 1–2 seconds.');
        }
      };

      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Recorder initialization failed';
      setCameraError(`Recording failed: ${message}`);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      setIsFinalizing(true);
      setIsRecording(false);
      mediaRecorderRef.current.stop();
    }
  };

  const isBusy = isRecording || isFinalizing || isUploading;

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 className="card-title">Camera Feed</h2>
        {isRecording && (
          <span className="recording-indicator">
            <span className="pulse-dot" /> Recording Segment...
          </span>
        )}
        {isFinalizing && (
          <span className="recording-indicator" style={{ color: 'var(--accent-amber)' }}>
            Finalizing Segment...
          </span>
        )}
      </div>

      <div className="camera-preview-container">
        {/* Video element stays permanently mounted in DOM so videoRef is always valid */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="camera-video"
          style={{ display: isCameraActive ? 'block' : 'none' }}
        />

        {!isCameraActive && (
          <div className="camera-placeholder">
            {cameraError ? (
              <p style={{ color: 'var(--accent-red)' }}>{cameraError}</p>
            ) : (
              <p>Camera is currently inactive. Click "Start Camera" to preview.</p>
            )}
          </div>
        )}
      </div>

      <div className="controls-row">
        {!isCameraActive ? (
          <button
            type="button"
            className="btn btn-primary"
            onClick={startCamera}
            disabled={isBusy}
          >
            Start Camera
          </button>
        ) : (
          <>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={stopCamera}
              disabled={isBusy}
            >
              Stop Camera
            </button>

            {!isRecording ? (
              <button
                type="button"
                className="btn btn-primary"
                onClick={startRecording}
                disabled={isBusy}
              >
                Record Sample (Start)
              </button>
            ) : (
              <button
                type="button"
                className="btn btn-danger"
                onClick={stopRecording}
                disabled={isFinalizing}
              >
                Stop &amp; Process Sample
              </button>
            )}
          </>
        )}
      </div>

      {cameraError && (
        <p style={{ color: 'var(--accent-red)', fontSize: '0.85rem', marginTop: '8px' }}>
          {cameraError}
        </p>
      )}
    </div>
  );
};
