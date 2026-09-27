import { useEffect, useRef, useState, useCallback } from 'react';
import type { UtteranceEvent, SpeakerInfo, VADState, LiveProsodyEvent, DebugMetrics, AccessibilitySettings } from '../types';

interface UseWebSocketOptions {
  onUtterance?: (utterance: UtteranceEvent) => void;
  onSpeakerList?: (speakers: SpeakerInfo[]) => void;
  onVADState?: (vad: VADState) => void;
  onProsodyState?: (prosody: LiveProsodyEvent) => void;
  onLiveStateReset?: () => void;
  onDebugMetrics?: (metrics: DebugMetrics) => void;
  onExportReady?: (data: { format: string; content: string }) => void;
  onDemoStatus?: (isRunning: boolean) => void;
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isDemoRunning, setIsDemoRunning] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const shouldReconnectRef = useRef(false);
  const optionsRef = useRef(options);

  useEffect(() => {
    optionsRef.current = options;
  }, [options]);

  const connect = useCallback(function connectSocket() {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // Use current host if running from FastAPI, or fallback to port 8000 in Vite dev mode
    const host = window.location.port === '5173' ? '127.0.0.1:8000' : window.location.host;
    const wsUrl = `${protocol}//${host}/ws/conversation`;

    try {
      const ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        setError(null);
      };

      ws.onclose = () => {
        optionsRef.current.onLiveStateReset?.();
        if (!shouldReconnectRef.current) return;
        setIsConnected(false);
        wsRef.current = null;
        // Attempt auto-reconnect after 2.5s
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connectSocket();
        }, 2500);
      };

      ws.onerror = () => {
        setError('Unable to connect to Hackcessible local server.');
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          const type = msg.type;
          const data = msg.data;

          if (type === 'utterance' && data && optionsRef.current.onUtterance) {
            optionsRef.current.onUtterance(data);
          } else if (type === 'speaker_list' && data && optionsRef.current.onSpeakerList) {
            optionsRef.current.onSpeakerList(data);
          } else if (type === 'vad_state' && data && optionsRef.current.onVADState) {
            optionsRef.current.onVADState(data);
          } else if (type === 'prosody_state' && data && optionsRef.current.onProsodyState) {
            optionsRef.current.onProsodyState(data);
          } else if (type === 'debug_metrics' && data && optionsRef.current.onDebugMetrics) {
            optionsRef.current.onDebugMetrics(data);
          } else if (type === 'export_ready' && data && optionsRef.current.onExportReady) {
            optionsRef.current.onExportReady(data);
          } else if (type === 'demo_started') {
            setIsDemoRunning(true);
            if (optionsRef.current.onDemoStatus) optionsRef.current.onDemoStatus(true);
          } else if (type === 'demo_stopped' || type === 'demo_completed') {
            setIsDemoRunning(false);
            if (optionsRef.current.onDemoStatus) optionsRef.current.onDemoStatus(false);
          } else if (type === 'session_state' && data) {
            optionsRef.current.onLiveStateReset?.();
            if (data.is_demo !== undefined) {
              setIsDemoRunning(data.is_demo);
              if (optionsRef.current.onDemoStatus) optionsRef.current.onDemoStatus(data.is_demo);
            }
            if (data.speakers && optionsRef.current.onSpeakerList) {
              optionsRef.current.onSpeakerList(data.speakers);
            }
          }
        } catch (e) {
          console.error('Failed to parse WebSocket message', e);
        }
      };
    } catch {
      console.error('WebSocket initialization failed.');
    }
  }, []);

  useEffect(() => {
    shouldReconnectRef.current = true;
    connect();
    return () => {
      shouldReconnectRef.current = false;
      if (reconnectTimeoutRef.current) {
        window.clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  const sendJson = useCallback((payload: Record<string, any>) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload));
    }
  }, []);

  const sendBinary = useCallback((data: ArrayBuffer) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    }
  }, []);

  const renameSpeaker = useCallback((speakerId: string, newName: string) => {
    sendJson({
      type: 'rename_speaker',
      speaker_id: speakerId,
      new_name: newName
    });
  }, [sendJson]);

  const startDemo = useCallback(() => {
    optionsRef.current.onLiveStateReset?.();
    sendJson({ type: 'start_demo' });
  }, [sendJson]);

  const stopDemo = useCallback(() => {
    sendJson({ type: 'stop_demo' });
  }, [sendJson]);

  const clearHistory = useCallback(() => {
    sendJson({ type: 'clear_history' });
  }, [sendJson]);

  const exportTranscript = useCallback((format: 'txt' | 'json' | 'vtt') => {
    sendJson({ type: 'export_transcript', format });
  }, [sendJson]);

  const updateSettings = useCallback((settings: AccessibilitySettings) => {
    sendJson({ type: 'update_settings', settings });
  }, [sendJson]);

  const startSession = useCallback((sessionId: string, settings: AccessibilitySettings) => {
    optionsRef.current.onLiveStateReset?.();
    sendJson({ type: 'start_session', session_id: sessionId, settings });
  }, [sendJson]);

  return {
    isConnected,
    isDemoRunning,
    error,
    sendBinary,
    renameSpeaker,
    startDemo,
    stopDemo,
    clearHistory,
    exportTranscript,
    updateSettings,
    startSession,
    reconnect: connect
  };
}
