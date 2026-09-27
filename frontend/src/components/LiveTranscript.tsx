import React, { useRef, useEffect, useState } from 'react';
import type { UtteranceEvent, AccessibilitySettings } from '../types';
import { UtteranceCard } from './UtteranceCard';
import { ArrowDown, MessageSquare, Mic, Play } from 'lucide-react';

interface LiveTranscriptProps {
  utterances: UtteranceEvent[];
  settings: AccessibilitySettings;
  onRenameSpeaker: (speakerId: string, currentLabel: string) => void;
  onStartMic: () => void;
  onStartDemo: () => void;
  isDemoRunning: boolean;
  isRecording: boolean;
}

export const LiveTranscript: React.FC<LiveTranscriptProps> = ({
  utterances,
  settings,
  onRenameSpeaker,
  onStartMic,
  onStartDemo,
  isDemoRunning,
  isRecording
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [lastReadUtteranceId, setLastReadUtteranceId] = useState<string | null>(null);
  const [isAutoScroll, setIsAutoScroll] = useState<boolean>(true);
  const lastReadIndex = lastReadUtteranceId
    ? utterances.findIndex((utterance) => utterance.id === lastReadUtteranceId)
    : -1;
  const unreadCount = isAutoScroll
    ? 0
    : lastReadIndex < 0
      ? utterances.length
      : Math.max(0, utterances.length - lastReadIndex - 1);

  // Handle user scroll detection
  const handleScroll = () => {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const atBottom = scrollHeight - scrollTop - clientHeight < 60;
    
    if (atBottom) {
      setLastReadUtteranceId(utterances.length ? utterances[utterances.length - 1].id : null);
      setIsAutoScroll(true);
    } else {
      setLastReadUtteranceId(utterances.length ? utterances[utterances.length - 1].id : null);
      setIsAutoScroll(false);
    }
  };

  // Scroll to bottom on new utterance if auto-scroll is enabled
  useEffect(() => {
    if (!containerRef.current) return;
    if (isAutoScroll) {
      if (typeof containerRef.current.scrollTo === 'function') {
        containerRef.current.scrollTo({
          top: containerRef.current.scrollHeight,
          behavior: settings.reduced_motion ? 'auto' : 'smooth'
        });
      } else {
        containerRef.current.scrollTop = containerRef.current.scrollHeight;
      }
    }
  }, [utterances, isAutoScroll, settings.reduced_motion]);

  const scrollToBottom = () => {
    if (!containerRef.current) return;
    if (typeof containerRef.current.scrollTo === 'function') {
      containerRef.current.scrollTo({
        top: containerRef.current.scrollHeight,
        behavior: 'smooth'
      });
    } else {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
    setLastReadUtteranceId(utterances.length ? utterances[utterances.length - 1].id : null);
    setIsAutoScroll(true);
  };

  return (
    <main
      className="transcript-container"
      ref={containerRef}
      onScroll={handleScroll}
      role="region"
      aria-label="Live conversation captions"
      tabIndex={0}
    >
      {utterances.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon-wrap">
            <MessageSquare size={44} className="empty-icon" />
          </div>
          <h2 className="empty-title">Ready for Multi-Speaker Conversation</h2>
          <p className="empty-desc">
            Standard live captions only show <em>what</em> was said. Hackcessible visually highlights{' '}
            <strong>who</strong> spoke, <strong>interrupting overlaps</strong>, and observable{' '}
            <strong>vocal inflections</strong> (rising pitch, fast speech, pauses) without guessing emotions.
          </p>

          <div className="empty-actions">
            <button
              onClick={onStartMic}
              className="btn-action btn-primary empty-btn"
              disabled={isDemoRunning}
            >
              <Mic size={18} />
              <span>Start Microphone</span>
            </button>
            <button
              onClick={onStartDemo}
              className="btn-action btn-secondary empty-btn"
              disabled={isRecording}
            >
              <Play size={18} />
              <span>Run 3-Speaker Demo</span>
            </button>
          </div>

          <div className="empty-feature-grid">
            <div className="feature-card">
              <h3>🗣️ Consistent Speaker Tracking</h3>
              <p>Persistent identities (Speaker A, B, C) with stable color coding and one-click renaming.</p>
            </div>
            <div className="feature-card">
              <h3>⚡ Overlapping Speech</h3>
              <p>Visual highlighting when multiple participants talk over each other in seminars or debates.</p>
            </div>
            <div className="feature-card">
              <h3>📈 Measurable Prosody</h3>
              <p>Objective acoustic cues: rising/falling pitch (↗/↘), vocal emphasis, pauses, and speech rate.</p>
            </div>
            <div className="feature-card">
              <h3>🔒 Local-First Privacy</h3>
              <p>Speech inference runs locally. Audio stays in memory while a segment is processed, then its buffer is cleared. Model startup may contact Hugging Face for model metadata or uncached files.</p>
            </div>
          </div>
        </div>
      ) : (
        <div className="transcript-feed">
          {utterances.map((item) => (
            <UtteranceCard
              key={item.id}
              utterance={item}
              settings={settings}
              onRenameSpeaker={onRenameSpeaker}
            />
          ))}
        </div>
      )}

      {/* Floating Jump-to-bottom button when scrolled up */}
      {!isAutoScroll && unreadCount > 0 && (
        <button
          onClick={scrollToBottom}
          className="jump-bottom-btn"
          aria-label={`Jump to latest captions (${unreadCount} new)`}
        >
          <ArrowDown size={16} />
          <span>Latest captions ({unreadCount} new)</span>
        </button>
      )}
    </main>
  );
};
