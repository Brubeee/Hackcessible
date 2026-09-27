import React from 'react';
import type { AccessibilitySettings } from '../types';
import { X, ShieldCheck, Download, Eye, Type, Sliders } from 'lucide-react';

interface SettingsModalProps {
  isOpen: boolean;
  settings: AccessibilitySettings;
  onClose: () => void;
  onUpdateSettings: (newSettings: AccessibilitySettings) => void;
  onExportTranscript: (format: 'txt' | 'json' | 'vtt') => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  settings,
  onClose,
  onUpdateSettings,
  onExportTranscript
}) => {
  if (!isOpen) return null;

  const handleToggle = (key: keyof AccessibilitySettings) => {
    const nextSettings = {
      ...settings,
      [key]: !settings[key]
    };
    if (key === 'privacy_mode' && nextSettings.privacy_mode) {
      nextSettings.save_transcript = false;
    }
    onUpdateSettings({
      ...nextSettings
    });
  };

  const handleFontSizeChange = (size: 'small' | 'medium' | 'large' | 'xlarge') => {
    onUpdateSettings({
      ...settings,
      font_size: size
    });
  };

  const handleContrastChange = (theme: 'normal' | 'high_contrast') => {
    onUpdateSettings({
      ...settings,
      contrast_theme: theme
    });
  };

  return (
    <div
      className="modal-overlay"
      role="presentation"
      onClick={onClose}
    >
      <div
        className="modal-card modal-large"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <div className="title-with-icon">
            <Sliders size={20} className="modal-title-icon" />
            <h2 id="settings-title" className="modal-title">
              Accessibility & Display Controls
            </h2>
          </div>
          <button
            onClick={onClose}
            className="modal-close-btn"
            aria-label="Close settings dialog"
          >
            <X size={18} />
          </button>
        </div>

        <div className="modal-body settings-scrollable">
          {/* Section 1: Accessibility Features */}
          <section className="settings-section">
            <h3 className="section-title">
              <Eye size={16} /> Visual Conversation Signals
            </h3>
            <p className="section-desc">
              Customizable visual layers designed so each user can focus on what helps them most.
            </p>

            <div className="toggle-list">
              <label className="toggle-item">
                <input
                  type="checkbox"
                  checked={settings.show_captions}
                  onChange={() => handleToggle('show_captions')}
                />
                <span className="toggle-label-wrap">
                  <span className="toggle-title">Live Captions</span>
                  <span className="toggle-subtitle">Display near-real-time speech-to-text text feed</span>
                </span>
              </label>

              <label className="toggle-item">
                <input
                  type="checkbox"
                  checked={settings.show_speaker_identities}
                  onChange={() => handleToggle('show_speaker_identities')}
                />
                <span className="toggle-label-wrap">
                  <span className="toggle-title">Speaker Diarization Identities</span>
                  <span className="toggle-subtitle">
                    Visually label who is speaking (Speaker A, B, C) with distinct accessible color accents
                  </span>
                </span>
              </label>

              <label className="toggle-item">
                <input
                  type="checkbox"
                  checked={settings.show_overlap_indicators}
                  onChange={() => handleToggle('show_overlap_indicators')}
                />
                <span className="toggle-label-wrap">
                  <span className="toggle-title">Simultaneous Overlap Badges</span>
                  <span className="toggle-subtitle">
                    Visually flag when multiple participants interrupt or speak concurrently
                  </span>
                </span>
              </label>

              <label className="toggle-item">
                <input
                  type="checkbox"
                  checked={settings.show_prosody_cues}
                  onChange={() => handleToggle('show_prosody_cues')}
                />
                <span className="toggle-label-wrap">
                  <span className="toggle-title">Measurable Prosody & Acoustic Cues</span>
                  <span className="toggle-subtitle">
                    Show rising/falling intonation (↗/↘), emphasis (●), speaking rate (»/‹), and pauses (⏸)
                  </span>
                </span>
              </label>

              <label className="toggle-item">
                <input
                  type="checkbox"
                  checked={settings.show_speaker_direction}
                  onChange={() => handleToggle('show_speaker_direction')}
                />
                <span className="toggle-label-wrap">
                  <span className="toggle-title">Spatial Speaker Direction (Simulated)</span>
                  <span className="toggle-subtitle">
                    Direction of Arrival indication (Left / Front / Right). Clearly labeled simulated in Demo.
                  </span>
                </span>
              </label>
            </div>
          </section>

          {/* Section 2: Typography & Sizing */}
          <section className="settings-section">
            <h3 className="section-title">
              <Type size={16} /> Caption Sizing & Readability
            </h3>

            <div className="setting-row">
              <span className="row-label">Caption Text Size:</span>
              <div className="btn-group-options" role="radiogroup" aria-label="Caption Text Size">
                {(['small', 'medium', 'large', 'xlarge'] as const).map((size) => (
                  <button
                    key={size}
                    type="button"
                    role="radio"
                    aria-checked={settings.font_size === size}
                    className={`btn-option ${settings.font_size === size ? 'active' : ''}`}
                    onClick={() => handleFontSizeChange(size)}
                  >
                    {size.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            <div className="setting-row">
              <span className="row-label">Contrast Theme:</span>
              <div className="btn-group-options" role="radiogroup" aria-label="Contrast Theme">
                <button
                  type="button"
                  role="radio"
                  aria-checked={settings.contrast_theme === 'normal'}
                  className={`btn-option ${settings.contrast_theme === 'normal' ? 'active' : ''}`}
                  onClick={() => handleContrastChange('normal')}
                >
                  Dark Slate (Standard)
                </button>
                <button
                  type="button"
                  role="radio"
                  aria-checked={settings.contrast_theme === 'high_contrast'}
                  className={`btn-option ${settings.contrast_theme === 'high_contrast' ? 'active' : ''}`}
                  onClick={() => handleContrastChange('high_contrast')}
                >
                  High Contrast
                </button>
              </div>
            </div>

            <div className="toggle-list compact-toggle-list">
              <label className="toggle-item">
                <input
                  type="checkbox"
                  checked={settings.reduced_motion}
                  onChange={() => handleToggle('reduced_motion')}
                />
                <span className="toggle-label-wrap">
                  <span className="toggle-title">Reduced Motion</span>
                  <span className="toggle-subtitle">Disable animated pulses and scrolling transitions</span>
                </span>
              </label>

              <label className="toggle-item">
                <input
                  type="checkbox"
                  checked={settings.compact_mode}
                  onChange={() => handleToggle('compact_mode')}
                />
                <span className="toggle-label-wrap">
                  <span className="toggle-title">Compact Card Layout</span>
                  <span className="toggle-subtitle">Condense spacing to fit more utterances on screen</span>
                </span>
              </label>
            </div>
          </section>

          {/* Section 3: Privacy & Data Retention */}
          <section className="settings-section">
            <h3 className="section-title">
              <ShieldCheck size={16} /> Privacy & Persistence
            </h3>

            <div className="privacy-card-info">
              <div className="privacy-card-title">
                <ShieldCheck size={18} className="shield-icon" />
                <span>Privacy Mode Enabled by Default</span>
              </div>
              <label className="toggle-item">
                <input
                  type="checkbox"
                  checked={settings.privacy_mode}
                  onChange={() => handleToggle('privacy_mode')}
                />
                <span className="toggle-label-wrap">
                  <span className="toggle-title">Privacy Mode: Block Transcript Disk Writes</span>
                  <span className="toggle-subtitle">
                    On by default. Turn this off before opting in to transcript persistence.
                  </span>
                </span>
              </label>
              <p className="privacy-card-text">
                Speech inference runs on this device. Audio stays in memory while its segment is
                processed, then the segment buffer is cleared; audio is not written to disk.
                Transcript saving is off by default. Model startup may contact Hugging Face for
                metadata or download model files that are not cached yet.
              </p>
            </div>

            <div className="toggle-list">
              <label className="toggle-item">
                <input
                  type="checkbox"
                  checked={settings.save_transcript}
                  disabled={settings.privacy_mode}
                  onChange={() => handleToggle('save_transcript')}
                />
                <span className="toggle-label-wrap">
                  <span className="toggle-title">Opt-In: Persist Transcript to Local SQLite</span>
                  <span className="toggle-subtitle">
                    OFF by default. Turn Privacy Mode off, then enable this to save text in local SQLite.
                  </span>
                </span>
              </label>
            </div>
          </section>

          {/* Section 4: Export Session */}
          <section className="settings-section">
            <h3 className="section-title">
              <Download size={16} /> Export Conversation
            </h3>
            <p className="section-desc">Download the current conversation session for review or sign language study.</p>

            <div className="export-btn-row">
              <button
                type="button"
                onClick={() => onExportTranscript('txt')}
                className="btn-action btn-secondary"
              >
                <Download size={15} />
                <span>Export Text (.txt)</span>
              </button>
              <button
                type="button"
                onClick={() => onExportTranscript('vtt')}
                className="btn-action btn-secondary"
              >
                <Download size={15} />
                <span>Export WebVTT (.vtt)</span>
              </button>
              <button
                type="button"
                onClick={() => onExportTranscript('json')}
                className="btn-action btn-secondary"
              >
                <Download size={15} />
                <span>Export JSON (.json)</span>
              </button>
            </div>
          </section>
        </div>

        <div className="modal-footer">
          <button
            type="button"
            onClick={onClose}
            className="btn-action btn-primary"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};
