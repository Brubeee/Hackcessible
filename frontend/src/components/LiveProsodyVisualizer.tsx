import React, { useMemo } from 'react';
import type { LiveProsodyEvent } from '../types';

interface LiveProsodyVisualizerProps {
  frames: LiveProsodyEvent[];
}

const WIDTH = 760;
const HEIGHT = 166;
const WINDOW_SECONDS = 20;

function makePath(
  frames: LiveProsodyEvent[],
  value: (frame: LiveProsodyEvent) => number | null,
  yFor: (value: number) => number,
  xFor: (frame: LiveProsodyEvent) => number,
): string {
  let path = '';
  let penDown = false;
  let previousTime = -Infinity;
  for (const frame of frames) {
    const sample = value(frame);
    if (sample === null || !Number.isFinite(sample) || frame.timestamp_s - previousTime > 0.7) {
      penDown = false;
      previousTime = frame.timestamp_s;
      continue;
    }
    const command = penDown ? 'L' : 'M';
    path += `${command}${xFor(frame).toFixed(1)},${yFor(sample).toFixed(1)} `;
    penDown = true;
    previousTime = frame.timestamp_s;
  }
  return path.trim();
}

function metric(value: number | null | undefined, digits = 0): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? '—'
    : value.toFixed(digits);
}

export const LiveProsodyVisualizer: React.FC<LiveProsodyVisualizerProps> = ({ frames }) => {
  const visibleFrames = useMemo(() => {
    if (!frames.length) return [];
    const cutoff = frames[frames.length - 1].timestamp_s - WINDOW_SECONDS;
    return frames.filter((frame) => frame.timestamp_s >= cutoff);
  }, [frames]);

  const latest = frames.length ? frames[frames.length - 1] : null;
  const xFor = (frame: LiveProsodyEvent) => {
    const end = latest?.timestamp_s ?? 0;
    return 44 + Math.max(0, Math.min(1, (frame.timestamp_s - (end - WINDOW_SECONDS)) / WINDOW_SECONDS)) * 704;
  };
  const pitchPath = makePath(
    visibleFrames,
    (frame) => frame.speech_active ? frame.pitch_hz : null,
    (pitch) => 54 - (Math.max(70, Math.min(400, pitch)) - 70) * 0.13,
    xFor,
  );
  const levelPath = makePath(
    visibleFrames,
    (frame) => frame.speech_active ? frame.level_dbfs : null,
    (dbfs) => 105 - (Math.max(-50, Math.min(-6, dbfs)) + 50) * 0.82,
    xFor,
  );
  const ratePath = makePath(
    visibleFrames,
    (frame) => frame.speech_active ? frame.speech_rate_wpm : null,
    (wpm) => 157 - Math.max(0, Math.min(300, wpm)) * 0.12,
    xFor,
  );

  return (
    <section className="live-prosody" aria-labelledby="live-prosody-heading">
      <div className="live-prosody-heading-row">
        <div>
          <h2 id="live-prosody-heading">Live vocal cues</h2>
          <p>Acoustic estimates from the microphone; these do not identify emotion or intent.</p>
        </div>
        <span className="prosody-live-status" aria-live="off">
          {latest ? (latest.speech_active ? 'Voice detected' : 'Listening / pause') : 'Waiting for measurements'}
        </span>
      </div>

      <div className="live-prosody-metrics" aria-label="Current acoustic measurements">
        <div className="live-prosody-metric">
          <span>Pitch (F0)</span>
          <strong>{latest?.speech_active && latest.pitch_hz !== null ? `${metric(latest.pitch_hz)} Hz` : '—'}</strong>
          <small>{latest?.speech_active
            ? latest.pitch_direction[0].toUpperCase() + latest.pitch_direction.slice(1)
            : 'No voiced estimate'}</small>
        </div>
        <div className="live-prosody-metric">
          <span>Microphone level</span>
          <strong>{latest ? `${metric(latest.level_dbfs, 1)} dBFS` : '—'}</strong>
          <small>{latest?.relative_level_db !== null && latest?.relative_level_db !== undefined
            ? `${latest.relative_level_db > 0 ? '+' : ''}${metric(latest.relative_level_db, 1)} dB vs recent speech`
            : 'Gain-dependent level'}</small>
        </div>
        <div className="live-prosody-metric">
          <span>Recognized pace</span>
          <strong>{latest?.speech_active && latest.speech_rate_wpm !== null ? `${metric(latest.speech_rate_wpm)} WPM` : '—'}</strong>
          <small>{latest?.speech_active && latest.speech_rate_wpm === null ? 'Waiting for recognized words' : 'Words per minute'}</small>
        </div>
        <div className="live-prosody-metric">
          <span>{latest?.speech_active ? 'Speech duration' : 'Pause'}</span>
          <strong>{latest?.speech_active
            ? `${metric(latest.speech_duration_s, 1)} s`
            : latest?.pause_duration_s !== null && latest?.pause_duration_s !== undefined
              ? `${metric(latest.pause_duration_s, 1)} s` : '—'}</strong>
          <small>{latest?.emphasis_candidate ? 'Possible level + pitch emphasis' : 'Speech timing cue'}</small>
        </div>
      </div>

      <div className="live-prosody-chart-wrap">
        {visibleFrames.length ? (
          <svg className="live-prosody-chart" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img"
            aria-label="Recent 20-second history of voiced pitch, microphone level, recognized speech pace, speech activity, and possible emphasis cues">
            <line x1="44" y1="58" x2="748" y2="58" className="prosody-grid" />
            <line x1="44" y1="109" x2="748" y2="109" className="prosody-grid" />
            <line x1="44" y1="160" x2="748" y2="160" className="prosody-grid" />
            <text x="2" y="18" className="prosody-axis-label">F0 Hz</text>
            <text x="2" y="73" className="prosody-axis-label">dBFS</text>
            <text x="2" y="126" className="prosody-axis-label">WPM</text>
            <path d={pitchPath} className="prosody-pitch-line" />
            <path d={levelPath} className="prosody-level-line" />
            <path d={ratePath} className="prosody-rate-line" />
            {visibleFrames.map((frame, index) => (
              <React.Fragment key={`${frame.timestamp_s}-${index}`}>
                {frame.speech_active && <rect x={xFor(frame) - 2} y="112" width="4" height="5" rx="1" className="prosody-speech-mark" />}
                {frame.emphasis_candidate && <path d={`M${xFor(frame) - 4},18 l4,-7 l4,7 z`} className="prosody-emphasis-mark" />}
              </React.Fragment>
            ))}
            <text x="44" y="165" className="prosody-axis-label">20 s ago</text>
            <text x="710" y="165" className="prosody-axis-label">now</text>
          </svg>
        ) : (
          <p className="live-prosody-empty">The chart fills as live microphone measurements arrive.</p>
        )}
      </div>
      <div className="live-prosody-legend" aria-label="Chart legend">
        <span><i className="legend-swatch pitch" />Pitch (Hz)</span>
        <span><i className="legend-swatch level" />Level (dBFS)</span>
        <span><i className="legend-swatch rate" />Recognized pace (WPM)</span>
        <span><i className="legend-swatch speech" />Speech-active frame</span>
        <span><i className="legend-swatch emphasis" />Possible emphasis</span>
      </div>
    </section>
  );
};
