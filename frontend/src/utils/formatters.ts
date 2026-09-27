import type { ProsodyFeatures } from '../types';

export function formatTime(seconds: number): string {
  if (isNaN(seconds) || seconds < 0) return '00:00';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

export interface ProsodyTag {
  id: string;
  icon: string;
  label: string;
  tooltip: string;
  badgeClass: string;
}

export function extractProsodyTags(prosody?: ProsodyFeatures): ProsodyTag[] {
  if (!prosody) return [];
  const tags: ProsodyTag[] = [];

  if (prosody.rising_intonation) {
    tags.push({
      id: 'rising',
      icon: '↗',
      label: 'Rising intonation',
      tooltip: 'Pitch contour rises significantly at end of phrase (often a question or check for confirmation)',
      badgeClass: 'badge-rising'
    });
  }

  if (prosody.falling_intonation) {
    tags.push({
      id: 'falling',
      icon: '↘',
      label: 'Falling intonation',
      tooltip: 'Pitch contour drops at end of phrase (often a definitive statement or conclusion)',
      badgeClass: 'badge-falling'
    });
  }

  if (prosody.emphasis) {
    tags.push({
      id: 'emphasis',
      icon: '●',
      label: 'Emphasis',
      tooltip: 'Acoustic vocal stress / local intensity burst',
      badgeClass: 'badge-emphasis'
    });
  }

  if (prosody.long_pause_before) {
    tags.push({
      id: 'pause',
      icon: '⏸',
      label: 'Long pause',
      tooltip: 'Preceded by noticeable conversational silence (>1.2s)',
      badgeClass: 'badge-pause'
    });
  }

  if (prosody.speech_rate === 'fast') {
    tags.push({
      id: 'fast',
      icon: '»',
      label: 'Fast speech',
      tooltip: 'Speaking rate is noticeably rapid (>215 words per minute)',
      badgeClass: 'badge-rate-fast'
    });
  } else if (prosody.speech_rate === 'slow') {
    tags.push({
      id: 'slow',
      icon: '‹',
      label: 'Slow speech',
      tooltip: 'Deliberate or slower speaking rate (<100 words per minute)',
      badgeClass: 'badge-rate-slow'
    });
  }

  if (prosody.relative_volume === 'loud') {
    tags.push({
      id: 'loud',
      icon: '🔊',
      label: 'Louder',
      tooltip: 'Volume is noticeably louder than this speaker’s recent baseline',
      badgeClass: 'badge-vol-loud'
    });
  } else if (prosody.relative_volume === 'quiet') {
    tags.push({
      id: 'quiet',
      icon: '🔉',
      label: 'Quieter',
      tooltip: 'Volume is noticeably softer than this speaker’s recent baseline',
      badgeClass: 'badge-vol-quiet'
    });
  }

  return tags;
}
