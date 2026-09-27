import React, { useState, useEffect, useRef } from 'react';
import { X, Check } from 'lucide-react';

interface RenameSpeakerModalProps {
  isOpen: boolean;
  speakerId: string | null;
  currentLabel: string;
  onClose: () => void;
  onSave: (speakerId: string, newName: string) => void;
}

export const RenameSpeakerModal: React.FC<RenameSpeakerModalProps> = ({
  isOpen,
  speakerId,
  currentLabel,
  onClose,
  onSave
}) => {
  const [name, setName] = useState(currentLabel);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!isOpen) return;

    const focusTimer = window.setTimeout(() => {
      inputRef.current?.focus();
      inputRef.current?.select();
    }, 50);
    return () => window.clearTimeout(focusTimer);
  }, [isOpen, currentLabel]);

  if (!isOpen || !speakerId) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (name.trim()) {
      onSave(speakerId, name.trim());
      onClose();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      onClose();
    }
  };

  return (
    <div
      className="modal-overlay"
      role="presentation"
      onClick={onClose}
      onKeyDown={handleKeyDown}
    >
      <div
        className="modal-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="rename-modal-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <h2 id="rename-modal-title" className="modal-title">
            Rename Speaker Identity
          </h2>
          <button
            onClick={onClose}
            className="modal-close-btn"
            aria-label="Close rename dialog"
          >
            <X size={18} />
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="modal-body">
            <p className="modal-intro">
              Assign a permanent name for this speaker in this session (e.g.{' '}
              <em>"Professor"</em>, <em>"Riya"</em>, <em>"Dr. Rao"</em>). The assigned color
              remains identical.
            </p>

            <label htmlFor="speaker-name-input" className="form-label">
              Speaker Name:
            </label>
            <input
              id="speaker-name-input"
              ref={inputRef}
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Professor"
              className="form-input"
              maxLength={40}
              required
            />
          </div>

          <div className="modal-footer">
            <button
              type="button"
              onClick={onClose}
              className="btn-action btn-secondary"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn-action btn-primary"
            >
              <Check size={16} />
              <span>Save Name</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
