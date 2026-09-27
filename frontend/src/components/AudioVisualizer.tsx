import React from 'react';
import { Mic } from 'lucide-react';

interface AudioVisualizerProps {
  level: number; // 0.0 to 1.0
  isSpeaking: boolean;
  isRecording: boolean;
}

export const AudioVisualizer: React.FC<AudioVisualizerProps> = ({
  level,
  isSpeaking,
  isRecording
}) => {
  if (!isRecording) return null;

  return (
    <div className="audio-meter-bar" title="Microphone Input Level">
      <Mic size={14} className={isSpeaking ? 'mic-icon active' : 'mic-icon'} />
      <div className="meter-track">
        <div
          className={`meter-bar ${isSpeaking ? 'speaking' : ''}`}
          style={{ width: `${Math.max(5, Math.min(100, Math.round(level * 100)))}%` }}
        />
      </div>
      <span className="meter-text">{isSpeaking ? 'Voice Active' : 'Listening'}</span>
    </div>
  );
};
