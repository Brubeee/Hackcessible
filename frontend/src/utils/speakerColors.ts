export interface SpeakerColorStyle {
  border: string;
  badgeBg: string;
  badgeText: string;
  cardBg: string;
  accent: string;
}

// 8 distinct high-contrast accessible color definitions
export const SPEAKER_PALETTES: SpeakerColorStyle[] = [
  // Speaker A - Warm Amber/Gold
  {
    border: '#d97706',
    badgeBg: '#b45309',
    badgeText: '#ffffff',
    cardBg: 'rgba(217, 119, 6, 0.08)',
    accent: '#f59e0b'
  },
  // Speaker B - Electric Cyan/Sky
  {
    border: '#0284c7',
    badgeBg: '#0369a1',
    badgeText: '#ffffff',
    cardBg: 'rgba(2, 132, 199, 0.08)',
    accent: '#38bdf8'
  },
  // Speaker C - Vivid Emerald
  {
    border: '#059669',
    badgeBg: '#047857',
    badgeText: '#ffffff',
    cardBg: 'rgba(5, 150, 105, 0.08)',
    accent: '#34d399'
  },
  // Speaker D - Deep Purple/Violet
  {
    border: '#7c3aed',
    badgeBg: '#6d28d9',
    badgeText: '#ffffff',
    cardBg: 'rgba(124, 58, 237, 0.08)',
    accent: '#a78bfa'
  },
  // Speaker E - Vibrant Rose
  {
    border: '#e11d48',
    badgeBg: '#be123c',
    badgeText: '#ffffff',
    cardBg: 'rgba(225, 29, 72, 0.08)',
    accent: '#fb7185'
  },
  // Speaker F - Indigo
  {
    border: '#4f46e5',
    badgeBg: '#4338ca',
    badgeText: '#ffffff',
    cardBg: 'rgba(79, 70, 229, 0.08)',
    accent: '#818cf8'
  },
  // Speaker G - Lime
  {
    border: '#65a30d',
    badgeBg: '#4d7c0f',
    badgeText: '#ffffff',
    cardBg: 'rgba(101, 163, 13, 0.08)',
    accent: '#a3e635'
  },
  // Speaker H - Coral/Orange
  {
    border: '#ea580c',
    badgeBg: '#c2410c',
    badgeText: '#ffffff',
    cardBg: 'rgba(234, 88, 12, 0.08)',
    accent: '#fb923c'
  }
];

export function getSpeakerStyle(colorIndex: number): SpeakerColorStyle {
  const safeIndex = Math.abs(colorIndex) % SPEAKER_PALETTES.length;
  return SPEAKER_PALETTES[safeIndex];
}
