import React from 'react';
import type { UtteranceEvent, AccessibilitySettings } from '../types';
import { getSpeakerStyle } from '../utils/speakerColors';
import { formatTime, extractProsodyTags } from '../utils/formatters';
import { Compass, Users, Edit3 } from 'lucide-react';

interface UtteranceCardProps {
  utterance: UtteranceEvent;
  settings: AccessibilitySettings;
  onRenameSpeaker: (speakerId: string, currentLabel: string) => void;
}

export const UtteranceCard: React.FC<UtteranceCardProps> = ({
  utterance,
  settings,
  onRenameSpeaker
}) => {
  const speakerStyle = getSpeakerStyle(utterance.speaker_color_index);
  const prosodyTags = extractProsodyTags(utterance.prosody);

  const cardStyle: React.CSSProperties = {
    borderLeft: `5px solid ${speakerStyle.border}`,
    backgroundColor: utterance.overlap ? 'rgba(239, 68, 68, 0.08)' : speakerStyle.cardBg
  };

  return (
    <article
      className={`utterance-card ${utterance.overlap ? 'overlap-active' : ''} ${!utterance.is_final ? 'interim-card' : ''}`}
      style={cardStyle}
      aria-labelledby={`speaker-${utterance.id}`}
    >
      <div className="card-header">
        <div className="speaker-meta">
          {settings.show_speaker_identities && (
            <button
              onClick={() => onRenameSpeaker(utterance.speaker_id, utterance.speaker_label)}
              className="speaker-badge-btn"
              style={{
                backgroundColor: speakerStyle.badgeBg,
                color: speakerStyle.badgeText
              }}
              title="Click to rename this speaker for this session"
              aria-label={`Speaker: ${utterance.speaker_label}. Click to rename.`}
            >
              <span id={`speaker-${utterance.id}`} className="speaker-name">
                {utterance.speaker_label}
              </span>
              <Edit3 size={11} className="rename-icon" />
            </button>
          )}

          <time className="utterance-time" dateTime={new Date().toISOString()}>
            {formatTime(utterance.start_time)}
          </time>
        </div>

        <div className="badges-meta">
          {/* Overlap Indicator */}
          {settings.show_overlap_indicators && utterance.overlap && (
            <div
              className="overlap-badge"
              role="alert"
              title={`Simultaneous overlapping speech with: ${utterance.overlapping_speakers.join(', ') || 'another speaker'}`}
            >
              <Users size={13} />
              <span>
                OVERLAP
                {utterance.overlapping_speakers.length > 0 &&
                  `: ${utterance.overlapping_speakers.join(' + ')}`}
              </span>
            </div>
          )}

          {/* Spatial Direction Indicator */}
          {settings.show_speaker_direction && utterance.direction && (
            <div
              className="direction-badge"
              title={`Speaker position: ${utterance.direction.label} (${utterance.direction.angle_degrees}°)${utterance.direction.simulated ? ' [Simulated]' : ''}`}
            >
              <Compass size={13} />
              <span>
                {utterance.direction.label}
                {utterance.direction.simulated && ' (Sim)'}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Caption Text */}
      {settings.show_captions && (
        <div className="card-body">
          <p className={`caption-text ${!utterance.is_final ? 'caption-interim' : ''}`}>
            {utterance.text}
            {!utterance.is_final && <span className="typing-cursor" aria-hidden="true" />}
          </p>
        </div>
      )}

      {/* Prosody / Acoustic Cues */}
      {settings.show_prosody_cues && utterance.is_final && (
        prosodyTags.length > 0 ||
        utterance.prosody.mean_pitch_hz != null ||
        utterance.prosody.min_pitch_hz != null ||
        utterance.prosody.words_per_minute != null ||
        utterance.prosody.rms_db != null
      ) && (
        <div className="card-prosody-row" aria-label="Acoustic vocal properties">
          {prosodyTags.map((tag) => (
            <span
              key={tag.id}
              className={`prosody-tag ${tag.badgeClass}`}
              title={tag.tooltip}
              aria-label={tag.label}
            >
              <span className="prosody-icon" aria-hidden="true">
                {tag.icon}
              </span>
              <span className="prosody-label">{tag.label}</span>
            </span>
          ))}

          {/* Measurable Pitch & RMS info */}
          {utterance.prosody.mean_pitch_hz && (
            <span
              className="prosody-stat"
              title={`Acoustic fundamental frequency: ${Math.round(utterance.prosody.mean_pitch_hz)} Hz`}
            >
              {Math.round(utterance.prosody.mean_pitch_hz)} Hz
            </span>
          )}
          {utterance.prosody.min_pitch_hz != null && utterance.prosody.max_pitch_hz != null && (
            <span
              className="prosody-stat"
              title={`Voiced fundamental-frequency range: ${Math.round(utterance.prosody.min_pitch_hz)} to ${Math.round(utterance.prosody.max_pitch_hz)} Hz`}
            >
              {Math.round(utterance.prosody.min_pitch_hz)}–{Math.round(utterance.prosody.max_pitch_hz)} Hz range
            </span>
          )}
          {utterance.prosody.pitch_range_semitones != null && (
            <span className="prosody-stat" title="Pitch range across voiced frames, in semitones">
              {utterance.prosody.pitch_range_semitones.toFixed(1)} ST range
            </span>
          )}
          {utterance.prosody.words_per_minute != null && (
            <span className="prosody-stat" title="Recognized words per minute over this utterance">
              {Math.round(utterance.prosody.words_per_minute)} WPM
            </span>
          )}
          {utterance.prosody.rms_db != null && (
            <span className="prosody-stat" title="Root mean square signal level in dBFS; depends on microphone gain">
              {utterance.prosody.rms_db.toFixed(1)} dBFS
            </span>
          )}
          {utterance.prosody.pitch_semitone_excursion !== null &&
           utterance.prosody.pitch_semitone_excursion !== undefined && (
            <span
              className="prosody-stat"
              title={`Pitch contour excursion: ${utterance.prosody.pitch_semitone_excursion > 0 ? '+' : ''}${utterance.prosody.pitch_semitone_excursion.toFixed(1)} semitones relative to syllable nucleus`}
            >
              {utterance.prosody.pitch_semitone_excursion > 0 ? '+' : ''}
              {utterance.prosody.pitch_semitone_excursion.toFixed(1)} ST
            </span>
          )}
        </div>
      )}
    </article>
  );
};
