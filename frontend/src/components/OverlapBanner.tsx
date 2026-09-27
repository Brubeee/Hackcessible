import React from 'react';
import { Users } from 'lucide-react';

interface OverlapBannerProps {
  speakers: string[];
}

export const OverlapBanner: React.FC<OverlapBannerProps> = ({ speakers }) => {
  if (speakers.length === 0) return null;

  return (
    <div className="overlap-active-banner" role="status" aria-live="assertive">
      <div className="banner-content">
        <Users size={16} className="overlap-banner-icon" />
        <span className="banner-title">SIMULTANEOUS SPEECH DETECTED:</span>
        <span className="banner-speakers">{speakers.join(' + ')}</span>
      </div>
    </div>
  );
};
