import { describe, it, expect, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { UtteranceCard } from '../components/UtteranceCard';
import { LiveTranscript } from '../components/LiveTranscript';
import { DevDebugDrawer } from '../components/DevDebugDrawer';
import { SettingsModal } from '../components/SettingsModal';
import { LiveProsodyVisualizer } from '../components/LiveProsodyVisualizer';
import { formatTime, extractProsodyTags } from '../utils/formatters';
import type { UtteranceEvent, AccessibilitySettings, DebugMetrics } from '../types';
import type { LiveProsodyEvent } from '../types';

const mockSettings: AccessibilitySettings = {
  show_captions: true,
  show_speaker_identities: true,
  show_overlap_indicators: true,
  show_prosody_cues: true,
  show_speaker_direction: true,
  save_transcript: false,
  privacy_mode: true,
  font_size: 'medium',
  contrast_theme: 'normal',
  reduced_motion: false,
  compact_mode: false,
  simulated_direction: true
};

const mockUtterance: UtteranceEvent = {
  id: 'utt-1',
  session_id: 'test_session',
  start_time: 32.5,
  end_time: 35.0,
  speaker_id: 'speaker_1',
  speaker_label: 'Dr. Rao (Speaker A)',
  speaker_color_index: 0,
  text: 'Could this be linked to social anxiety?',
  is_final: true,
  overlap: false,
  overlapping_speakers: [],
  prosody: {
    rising_intonation: true,
    falling_intonation: false,
    emphasis: true,
    long_pause_before: false,
    speech_rate: 'fast',
    relative_volume: 'loud',
    mean_pitch_hz: 210.0,
    rms_db: -18.5,
    words_per_minute: 220.0
  },
  direction: {
    angle_degrees: -45.0,
    label: 'left',
    simulated: true,
    confidence: 0.9
  },
  confidence: {
    transcription: 0.95,
    speaker: 0.9
  }
};

describe('Accessibility & Formatters', () => {
  it('formats time accurately into mm:ss', () => {
    expect(formatTime(0)).toBe('00:00');
    expect(formatTime(65)).toBe('01:05');
    expect(formatTime(125.7)).toBe('02:05');
  });

  it('extracts neutral prosody tags without emotional inference', () => {
    const tags = extractProsodyTags(mockUtterance.prosody);
    const labels = tags.map((t) => t.label);
    expect(labels).toContain('Rising intonation');
    expect(labels).toContain('Emphasis');
    expect(labels).toContain('Fast speech');
    expect(labels).toContain('Louder');
    // Ensure no psychological emotion labels are generated
    expect(labels).not.toContain('Angry');
    expect(labels).not.toContain('Anxious');
    expect(labels).not.toContain('Happy');
  });
});

describe('LiveProsodyVisualizer', () => {
  const frame: LiveProsodyEvent = {
    timestamp_s: 4.8,
    speech_active: true,
    pitch_hz: 216,
    pitch_direction: 'rising',
    level_dbfs: -18.4,
    relative_level_db: 2.3,
    speech_rate_wpm: 174,
    speech_duration_s: 2.2,
    pause_duration_s: 0,
    emphasis_candidate: true,
  };

  it('renders measured vocal features, units, legend, and neutral safety copy', () => {
    render(<LiveProsodyVisualizer frames={[{ ...frame, timestamp_s: 4.6 }, frame]} />);
    expect(screen.getByRole('region', { name: 'Live vocal cues' })).toBeDefined();
    expect(screen.getByText('216 Hz')).toBeDefined();
    expect(screen.getByText('Rising')).toBeDefined();
    expect(screen.getByText('174 WPM')).toBeDefined();
    expect(screen.getByText('-18.4 dBFS')).toBeDefined();
    expect(screen.getByText('Speech duration')).toBeDefined();
    expect(screen.getByText('2.2 s')).toBeDefined();
    expect(screen.getByText('Possible level + pitch emphasis')).toBeDefined();
    expect(screen.getByRole('img', { name: /Recent 20-second history/ })).toBeDefined();
    expect(screen.getByText(/do not identify emotion or intent/)).toBeDefined();
    expect(screen.queryByText(/angry|happy|anxious/i)).toBeNull();
  });

  it('shows unavailable pitch and pace honestly when no voice is detected', () => {
    render(<LiveProsodyVisualizer frames={[{
      ...frame,
      speech_active: false,
      pitch_hz: null,
      pitch_direction: 'unavailable',
      speech_rate_wpm: null,
      pause_duration_s: 1.4,
      emphasis_candidate: false,
    }]} />);
    expect(screen.getByText('Listening / pause')).toBeDefined();
    expect(screen.getByText('Pause')).toBeDefined();
    expect(screen.getAllByText('—').length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText('1.4 s')).toBeDefined();
  });

  it('provides an empty-state explanation before the first measurement', () => {
    render(<LiveProsodyVisualizer frames={[]} />);
    expect(screen.getByText('Waiting for measurements')).toBeDefined();
    expect(screen.getByText(/chart fills as live microphone measurements arrive/i)).toBeDefined();
  });
});

describe('UtteranceCard Component', () => {
  it('renders speaker label and caption text with accessible attributes', () => {
    const onRename = vi.fn();
    render(
      <UtteranceCard
        utterance={mockUtterance}
        settings={mockSettings}
        onRenameSpeaker={onRename}
      />
    );

    expect(screen.getByText('Dr. Rao (Speaker A)')).toBeDefined();
    expect(screen.getByText('Could this be linked to social anxiety?')).toBeDefined();
    expect(screen.getByText('Rising intonation')).toBeDefined();
  });

  it('renders overlap alert when speech overlap is detected', () => {
    const overlapUtterance: UtteranceEvent = {
      ...mockUtterance,
      overlap: true,
      overlapping_speakers: ['Priya (Speaker B)']
    };
    render(
      <UtteranceCard
        utterance={overlapUtterance}
        settings={mockSettings}
        onRenameSpeaker={vi.fn()}
      />
    );

    const overlapBadge = screen.getByRole('alert');
    expect(overlapBadge.textContent).toContain('OVERLAP');
    expect(overlapBadge.textContent).toContain('Priya (Speaker B)');
  });
});

describe('LiveTranscript Component', () => {
  it('renders empty state with Start Mic and Play Demo buttons when no utterances exist', () => {
    render(
      <LiveTranscript
        utterances={[]}
        settings={mockSettings}
        onRenameSpeaker={vi.fn()}
        onStartMic={vi.fn()}
        onStartDemo={vi.fn()}
        isDemoRunning={false}
        isRecording={false}
      />
    );

    expect(screen.getByText('Ready for Multi-Speaker Conversation')).toBeDefined();
    expect(screen.getByText('Start Microphone')).toBeDefined();
    expect(screen.getByText('Run 3-Speaker Demo')).toBeDefined();
  });
});

describe('Developer diagnostics', () => {
  const baseMetrics: DebugMetrics = {
    chunk_duration_ms: 1600,
    vad_latency_ms: 4,
    asr_latency_ms: 78,
    diarization_latency_ms: 14,
    prosody_latency_ms: 6,
    pipeline_processing_ms: 103,
    active_speakers_count: 3,
    queue_backlog: 0,
    simulated: false
  };

  const renderDrawer = (metrics: DebugMetrics) => render(
    <DevDebugDrawer
      isOpen
      onClose={vi.fn()}
      metrics={metrics}
      speakers={[]}
      vadState={null}
      isConnected
      audioLevel={0}
    />
  );

  it('does not present demo timing values as measured performance', () => {
    renderDrawer({ ...baseMetrics, simulated: true });

    expect(screen.getByRole('status').textContent).toContain('simulated');
    const metricSection = screen.getByText('Pipeline Processing Time').closest('section');
    const metricValues = metricSection?.querySelectorAll('.metric-val');
    expect(Array.from(metricValues ?? []).map((node) => node.textContent)).toEqual(['--', '--', '--', '--']);
    expect(screen.queryByText('103 ms')).toBeNull();
  });

  it('shows measured pipeline timing when metrics came from live processing', () => {
    renderDrawer(baseMetrics);

    expect(screen.getByText('103 ms')).toBeDefined();
    expect(screen.queryByRole('status')).toBeNull();
  });
});

describe('Privacy settings', () => {
  it('lets users explicitly disable disk-blocking mode before enabling transcript saves', () => {
    const onUpdateSettings = vi.fn();
    render(
      <SettingsModal
        isOpen
        settings={mockSettings}
        onClose={vi.fn()}
        onUpdateSettings={onUpdateSettings}
        onExportTranscript={vi.fn()}
      />
    );

    const privacyMode = screen.getByLabelText(/Privacy Mode: Block Transcript Disk Writes/);
    const saveTranscript = screen.getByLabelText(/Opt-In: Persist Transcript to Local SQLite/);
    expect((saveTranscript as HTMLInputElement).disabled).toBe(true);
    fireEvent.click(privacyMode);
    expect(onUpdateSettings).toHaveBeenCalledWith({ ...mockSettings, privacy_mode: false });
  });
});
