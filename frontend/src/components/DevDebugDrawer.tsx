import React from 'react';
import type { DebugMetrics, SpeakerInfo, VADState } from '../types';
import { Activity, Clock, Cpu, Server, X, UserCheck } from 'lucide-react';

interface DevDebugDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  metrics: DebugMetrics | null;
  speakers: SpeakerInfo[];
  vadState: VADState | null;
  isConnected: boolean;
  audioLevel: number;
}

export const DevDebugDrawer: React.FC<DevDebugDrawerProps> = ({
  isOpen,
  onClose,
  metrics,
  speakers,
  vadState,
  isConnected,
  audioLevel
}) => {
  if (!isOpen) return null;
  const showMeasuredMetrics = metrics !== null && !metrics.simulated;

  return (
    <aside className="debug-drawer" role="complementary" aria-label="Developer Diagnostics Panel">
      <div className="debug-header">
        <div className="title-with-icon">
          <Activity size={18} className="debug-icon" />
          <h2 className="debug-title">Diagnostics & Performance</h2>
        </div>
        <button
          onClick={onClose}
          className="modal-close-btn"
          aria-label="Close diagnostics drawer"
        >
          <X size={16} />
        </button>
      </div>

      <div className="debug-body">
        {/* Latency & Processing Metrics */}
        <section className="debug-section">
          <h3 className="debug-section-title">
            <Clock size={14} /> Pipeline Processing Time
          </h3>
          {metrics?.simulated && (
            <p className="text-muted" role="status">
              Demo metrics are simulated and do not measure live processing.
            </p>
          )}
          <div className="metrics-grid">
            <div className="metric-box">
              <span className="metric-val">
                {showMeasuredMetrics ? `${Math.round(metrics.pipeline_processing_ms)} ms` : '--'}
              </span>
              <span className="metric-name">Total Pipeline Compute</span>
            </div>
            <div className="metric-box">
              <span className="metric-val">
                {showMeasuredMetrics ? `${Math.round(metrics.asr_latency_ms)} ms` : '--'}
              </span>
              <span className="metric-name">Faster-Whisper ASR</span>
            </div>
            <div className="metric-box">
              <span className="metric-val">
                {showMeasuredMetrics ? `${Math.round(metrics.diarization_latency_ms)} ms` : '--'}
              </span>
              <span className="metric-name">Acoustic Diarization</span>
            </div>
            <div className="metric-box">
              <span className="metric-val">
                {showMeasuredMetrics ? `${Math.round(metrics.prosody_latency_ms)} ms` : '--'}
              </span>
              <span className="metric-name">Prosody Signal Proc</span>
            </div>
          </div>
        </section>

        {/* Real-time Audio & VAD */}
        <section className="debug-section">
          <h3 className="debug-section-title">
            <Cpu size={14} /> Signal Processing & VAD
          </h3>
          <div className="debug-list">
            <div className="debug-row">
              <span className="debug-k">Mic Signal Level:</span>
              <div className="gauge-wrap">
                <div
                  className="gauge-fill"
                  style={{ width: `${Math.round(audioLevel * 100)}%` }}
                />
              </div>
              <span className="debug-v">{Math.round(audioLevel * 100)}%</span>
            </div>

            <div className="debug-row">
              <span className="debug-k">Silero VAD Prob:</span>
              <div className="gauge-wrap">
                <div
                  className="gauge-fill vad-fill"
                  style={{
                    width: `${Math.round((vadState?.probability ?? 0) * 100)}%`
                  }}
                />
              </div>
              <span className="debug-v">
                {vadState ? `${(vadState.probability * 100).toFixed(1)}%` : '--'}
              </span>
            </div>

            <div className="debug-row">
              <span className="debug-k">VAD Active Speaking:</span>
              <span className={`debug-status-pill ${vadState?.speaking ? 'active' : ''}`}>
                {vadState?.speaking ? 'SPEECH DETECTED' : 'SILENCE'}
              </span>
            </div>

            <div className="debug-row">
              <span className="debug-k">Audio Energy (dBFS):</span>
              <span className="debug-v">
                {vadState ? `${vadState.energy.toFixed(1)} dB` : '--'}
              </span>
            </div>
          </div>
        </section>

        {/* Active Speaker Registry */}
        <section className="debug-section">
          <h3 className="debug-section-title">
            <UserCheck size={14} /> Session Speaker Registry ({speakers.length})
          </h3>
          <div className="debug-speaker-list">
            {speakers.map((spk) => (
              <div key={spk.id} className="debug-spk-row">
                <span className="spk-id-tag">{spk.id}</span>
                <span className="spk-label-val">{spk.label}</span>
                <span className="spk-cnt-val">{spk.total_utterances} utts</span>
              </div>
            ))}
          </div>
        </section>

        {/* Server & Connection */}
        <section className="debug-section">
          <h3 className="debug-section-title">
            <Server size={14} /> Pipeline Architecture
          </h3>
          <div className="debug-list text-muted">
            <div className="debug-row">
              <span className="debug-k">WebSocket Status:</span>
              <span className="debug-v">{isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
            <div className="debug-row">
              <span className="debug-k">ASR Engine:</span>
              <span className="debug-v">Faster-Whisper INT8 (Local CPU)</span>
            </div>
            <div className="debug-row">
              <span className="debug-k">VAD Engine:</span>
              <span className="debug-v">Silero VAD ONNX</span>
            </div>
            <div className="debug-row">
              <span className="debug-k">Diarizer:</span>
              <span className="debug-v">ECAPA-TDNN (MFCC fallback)</span>
            </div>
          </div>
        </section>
      </div>
    </aside>
  );
};
