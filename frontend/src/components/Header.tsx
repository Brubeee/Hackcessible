import React from 'react';
import { Mic, MicOff, Play, Square, Settings, ShieldCheck, Terminal, Trash2, WifiOff } from 'lucide-react';
import type { AccessibilitySettings } from '../types';

interface HeaderProps {
  isConnected: boolean;
  isRecording: boolean;
  isDemoRunning: boolean;
  settings: AccessibilitySettings;
  onToggleRecord: () => void;
  onToggleDemo: () => void;
  onOpenSettings: () => void;
  onToggleDebug: () => void;
  onClearHistory: () => void;
  isDebugOpen: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  isConnected,
  isRecording,
  isDemoRunning,
  settings,
  onToggleRecord,
  onToggleDemo,
  onOpenSettings,
  onToggleDebug,
  onClearHistory,
  isDebugOpen
}) => {
  return (
    <header className="header-container" role="banner">
      <div className="header-left">
        <div className="logo-group">
          <h1 className="logo-title">Hackcessible</h1>
          <span className="logo-badge" title="Hackcessible India 2026 Submission">
            India 2026
          </span>
        </div>
        <p className="logo-tagline">Visual Conversation Layer for Multi-Speaker Accessibility</p>
      </div>

      <div className="header-center">
        {/* Status Indicators */}
        <div className="status-pill" role="status" aria-live="polite">
          {isDemoRunning ? (
            <span className="pill-badge demo">
              <span className="status-dot demo-pulse" />
              DEMO MODE ACTIVE
            </span>
          ) : isRecording ? (
            <span className="pill-badge live">
              <span className="status-dot live-pulse" />
              LIVE CAPTIONING
            </span>
          ) : isConnected ? (
            <span className="pill-badge idle">
              <span className="status-dot idle-dot" />
              READY (STANDBY)
            </span>
          ) : (
            <span className="pill-badge offline">
              <WifiOff size={14} />
              DISCONNECTED
            </span>
          )}

          {/* Privacy Badge */}
          {settings.privacy_mode && (
            <div
              className="privacy-badge"
              title="Privacy Mode blocks transcript writes to disk. Speech inference runs locally. First startup may contact Hugging Face for model metadata or uncached model files."
            >
              <ShieldCheck size={15} className="privacy-icon" />
              <span>Privacy Active</span>
            </div>
          )}
        </div>
      </div>

      <div className="header-right">
        {/* Record Button */}
        <button
          onClick={onToggleRecord}
          disabled={!isConnected || isDemoRunning}
          className={`btn-action ${isRecording ? 'btn-danger' : 'btn-primary'}`}
          aria-label={isRecording ? 'Stop microphone capture' : 'Start microphone capture'}
          title={isRecording ? 'Stop listening' : 'Start microphone'}
        >
          {isRecording ? <MicOff size={18} /> : <Mic size={18} />}
          <span>{isRecording ? 'Stop Mic' : 'Start Mic'}</span>
        </button>

        {/* Demo Mode Button */}
        <button
          onClick={onToggleDemo}
          disabled={!isConnected || isRecording}
          className={`btn-action ${isDemoRunning ? 'btn-warning' : 'btn-secondary'}`}
          aria-label={isDemoRunning ? 'Stop 3-speaker demo conversation' : 'Start 3-speaker demo conversation'}
          title="Replay 3-speaker seminar demo"
        >
          {isDemoRunning ? <Square size={16} /> : <Play size={16} />}
          <span>{isDemoRunning ? 'Stop Demo' : 'Play Demo'}</span>
        </button>

        {/* Clear History */}
        <button
          onClick={onClearHistory}
          className="btn-icon"
          title="Clear Conversation History"
          aria-label="Clear Conversation History"
        >
          <Trash2 size={18} />
        </button>

        {/* Dev Debug Toggle */}
        <button
          onClick={onToggleDebug}
          className={`btn-icon ${isDebugOpen ? 'active' : ''}`}
          title="Toggle Diagnostics / Debug Drawer"
          aria-label="Toggle Diagnostics Drawer"
        >
          <Terminal size={18} />
        </button>

        {/* Settings Button */}
        <button
          onClick={onOpenSettings}
          className="btn-icon"
          title="Accessibility & Display Settings"
          aria-label="Open Settings"
        >
          <Settings size={18} />
        </button>
      </div>
    </header>
  );
};
