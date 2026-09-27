import React, { useState, useEffect, useCallback } from 'react';
import type { UtteranceEvent, SpeakerInfo, VADState, LiveProsodyEvent, DebugMetrics, AccessibilitySettings } from './types';
import { useWebSocket } from './hooks/useWebSocket';
import { useAudioCapture } from './hooks/useAudioCapture';
import { Header } from './components/Header';
import { LiveTranscript } from './components/LiveTranscript';
import { RenameSpeakerModal } from './components/RenameSpeakerModal';
import { SettingsModal } from './components/SettingsModal';
import { DevDebugDrawer } from './components/DevDebugDrawer';
import { OverlapBanner } from './components/OverlapBanner';
import { AudioVisualizer } from './components/AudioVisualizer';
import { LiveProsodyVisualizer } from './components/LiveProsodyVisualizer';
import { AlertCircle, X } from 'lucide-react';
import './styles/index.css';

const DEFAULT_SETTINGS: AccessibilitySettings = {
  show_captions: true,
  show_speaker_identities: true,
  show_overlap_indicators: true,
  show_prosody_cues: true,
  show_speaker_direction: false,
  save_transcript: false,
  privacy_mode: true,
  font_size: 'medium',
  contrast_theme: 'normal',
  reduced_motion: false,
  compact_mode: false,
  simulated_direction: true
};

export const App: React.FC = () => {
  const [utterances, setUtterances] = useState<UtteranceEvent[]>([]);
  const [speakers, setSpeakers] = useState<SpeakerInfo[]>([]);
  const [vadState, setVadState] = useState<VADState | null>(null);
  const [liveProsody, setLiveProsody] = useState<LiveProsodyEvent[]>([]);
  const [debugMetrics, setDebugMetrics] = useState<DebugMetrics | null>(null);
  const [activeOverlapSpeakers, setActiveOverlapSpeakers] = useState<string[]>([]);

  // Modals and Drawers
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isDebugOpen, setIsDebugOpen] = useState(false);
  const [renameData, setRenameData] = useState<{
    isOpen: boolean;
    speakerId: string | null;
    currentLabel: string;
  }>({
    isOpen: false,
    speakerId: null,
    currentLabel: ''
  });

  // Settings State (loaded from localStorage if present)
  const [settings, setSettings] = useState<AccessibilitySettings>(() => {
    try {
      const saved = localStorage.getItem('hackcessible_settings');
      return saved ? { ...DEFAULT_SETTINGS, ...JSON.parse(saved) } : DEFAULT_SETTINGS;
    } catch {
      return DEFAULT_SETTINGS;
    }
  });

  // Save settings changes
  const handleUpdateSettings = (newSettings: AccessibilitySettings) => {
    setSettings(newSettings);
    try {
      localStorage.setItem('hackcessible_settings', JSON.stringify(newSettings));
    } catch (e) {
      console.warn('Failed to save settings to localStorage', e);
    }
    ws.updateSettings(newSettings);
  };

  // Handle incoming utterance (both interim rolling and final)
  const handleUtterance = useCallback((newUtterance: UtteranceEvent) => {
    setUtterances((prev) => {
      const idx = prev.findIndex((u) => u.id === newUtterance.id);
      if (idx !== -1) {
        const copy = [...prev];
        copy[idx] = newUtterance;
        return copy;
      }
      return [...prev, newUtterance];
    });

    if (newUtterance.overlap) {
      const combined = [newUtterance.speaker_label, ...newUtterance.overlapping_speakers];
      setActiveOverlapSpeakers(Array.from(new Set(combined)));
      // Clear banner after 3.5 seconds
      setTimeout(() => {
        setActiveOverlapSpeakers([]);
      }, 3500);
    }
  }, []);

  const handleProsodyState = useCallback((frame: LiveProsodyEvent) => {
    setLiveProsody((previous) => [...previous, frame].slice(-150));
  }, []);

  // Handle export download
  const handleExportReady = useCallback((data: { format: string; content: string }) => {
    const mimeTypes: Record<string, string> = {
      txt: 'text/plain',
      json: 'application/json',
      vtt: 'text/vtt'
    };
    const blob = new Blob([data.content], { type: mimeTypes[data.format] || 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `hackcessible_transcript_${new Date().toISOString().slice(0, 10)}.${data.format}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, []);

  // WebSocket hook
  const ws = useWebSocket({
    onUtterance: handleUtterance,
    onSpeakerList: (spkList) => {
      setSpeakers(spkList);
      // Synchronize speaker labels in already-rendered utterances
      setUtterances((prev) =>
        prev.map((utt) => {
          const matched = spkList.find((s) => s.id === utt.speaker_id);
          if (matched && matched.label !== utt.speaker_label) {
            return { ...utt, speaker_label: matched.label };
          }
          return utt;
        })
      );
    },
    onVADState: setVadState,
    onProsodyState: handleProsodyState,
    onLiveStateReset: () => setLiveProsody([]),
    onDebugMetrics: setDebugMetrics,
    onExportReady: handleExportReady
  });

  // Audio capture hook
  const audio = useAudioCapture({
    onAudioChunk: (chunk) => {
      ws.sendBinary(chunk);
    }
  });

  // Sync DOM attributes for themes and font sizing
  useEffect(() => {
    document.body.setAttribute('data-theme', settings.contrast_theme);
    document.body.setAttribute('data-font-size', settings.font_size);
    document.body.setAttribute('data-compact', String(settings.compact_mode));
    if (settings.reduced_motion) {
      document.body.setAttribute('data-reduced-motion', 'true');
    } else {
      document.body.removeAttribute('data-reduced-motion');
    }
  }, [settings]);

  // Keyboard accessibility shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'm') {
        e.preventDefault();
        if (audio.isRecording) {
          audio.stopRecording();
        } else if (ws.isConnected) {
          if (ws.isDemoRunning) ws.stopDemo();
          ws.startSession(`live_${Date.now()}`, settings);
          setUtterances([]);
          setActiveOverlapSpeakers([]);
          setDebugMetrics(null);
          setVadState(null);
          setLiveProsody([]);
          audio.startRecording();
        }
      }
      if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
        e.preventDefault();
        if (ws.isDemoRunning) {
          ws.stopDemo();
        } else if (ws.isConnected) {
          if (audio.isRecording) audio.stopRecording();
          setUtterances([]);
          setActiveOverlapSpeakers([]);
          setDebugMetrics(null);
          setLiveProsody([]);
          ws.startDemo();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [audio, settings, ws]);

  const handleToggleRecord = async () => {
    if (audio.isRecording) {
      audio.stopRecording();
      setLiveProsody([]);
    } else {
      if (!ws.isConnected) return;
      if (ws.isDemoRunning) {
        ws.stopDemo();
      }
      ws.startSession(`live_${Date.now()}`, settings);
      setUtterances([]);
      setActiveOverlapSpeakers([]);
      setDebugMetrics(null);
      setVadState(null);
      setLiveProsody([]);
      await audio.startRecording();
    }
  };

  const handleToggleDemo = () => {
    if (ws.isDemoRunning) {
      ws.stopDemo();
    } else {
      if (audio.isRecording) {
        audio.stopRecording();
      }
      setUtterances([]);
      setActiveOverlapSpeakers([]);
      setDebugMetrics(null);
      setLiveProsody([]);
      ws.startDemo();
    }
  };

  const handleClearHistory = () => {
    setUtterances([]);
    setLiveProsody([]);
    ws.clearHistory();
  };

  const handleOpenRename = (speakerId: string, currentLabel: string) => {
    setRenameData({
      isOpen: true,
      speakerId,
      currentLabel
    });
  };

  const handleSaveRename = (speakerId: string, newName: string) => {
    ws.renameSpeaker(speakerId, newName);
  };

  return (
    <div className="app-container">
      {/* Top Header */}
      <Header
        isConnected={ws.isConnected}
        isRecording={audio.isRecording}
        isDemoRunning={ws.isDemoRunning}
        settings={settings}
        onToggleRecord={handleToggleRecord}
        onToggleDemo={handleToggleDemo}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onToggleDebug={() => setIsDebugOpen((prev) => !prev)}
        onClearHistory={handleClearHistory}
        isDebugOpen={isDebugOpen}
      />

      {/* Microphone Audio Level Bar */}
      <AudioVisualizer
        level={audio.audioLevel}
        isSpeaking={vadState?.speaking ?? false}
        isRecording={audio.isRecording}
      />

      {(audio.isRecording || liveProsody.length > 0) && !ws.isDemoRunning && settings.show_prosody_cues && (
        <LiveProsodyVisualizer frames={liveProsody} />
      )}

      {/* Microphone or Hardware Error Banner */}
      {audio.micError && (
        <div className="error-banner" role="alert">
          <div className="title-with-icon">
            <AlertCircle size={16} />
            <span>{audio.micError}</span>
          </div>
          <button
            onClick={audio.clearError}
            className="error-banner-close"
            aria-label="Dismiss error banner"
          >
            <X size={16} />
          </button>
        </div>
      )}

      {/* WebSocket Connection Error */}
      {!ws.isConnected && (
        <div className="error-banner" role="alert">
          <div className="title-with-icon">
            <AlertCircle size={16} />
            <span>Connecting to Hackcessible local Python backend on 127.0.0.1:8000...</span>
          </div>
        </div>
      )}

      {/* Simultaneous Overlap Banner */}
      {settings.show_overlap_indicators && (
        <OverlapBanner speakers={activeOverlapSpeakers} />
      )}

      {/* Main Conversation Captions Feed */}
      <LiveTranscript
        utterances={utterances}
        settings={settings}
        onRenameSpeaker={handleOpenRename}
        onStartMic={handleToggleRecord}
        onStartDemo={handleToggleDemo}
        isDemoRunning={ws.isDemoRunning}
        isRecording={audio.isRecording}
      />

      {/* Rename Speaker Modal */}
      <RenameSpeakerModal
        key={`${renameData.isOpen}:${renameData.speakerId ?? ''}:${renameData.currentLabel}`}
        isOpen={renameData.isOpen}
        speakerId={renameData.speakerId}
        currentLabel={renameData.currentLabel}
        onClose={() => setRenameData((prev) => ({ ...prev, isOpen: false }))}
        onSave={handleSaveRename}
      />

      {/* Settings & Accessibility Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        settings={settings}
        onClose={() => setIsSettingsOpen(false)}
        onUpdateSettings={handleUpdateSettings}
        onExportTranscript={ws.exportTranscript}
      />

      {/* Developer Diagnostics Drawer */}
      <DevDebugDrawer
        isOpen={isDebugOpen}
        onClose={() => setIsDebugOpen(false)}
        metrics={debugMetrics}
        speakers={speakers}
        vadState={vadState}
        isConnected={ws.isConnected}
        audioLevel={audio.audioLevel}
      />
    </div>
  );
};
export default App;
