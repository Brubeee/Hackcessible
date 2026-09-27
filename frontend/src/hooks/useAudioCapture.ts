import { useState, useRef, useCallback, useEffect } from 'react';

interface UseAudioCaptureOptions {
  onAudioChunk?: (pcmBytes: ArrayBuffer) => void;
  sampleRate?: number;
}

export function useAudioCapture({ onAudioChunk, sampleRate = 16000 }: UseAudioCaptureOptions = {}) {
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [micError, setMicError] = useState<string | null>(null);
  const [audioLevel, setAudioLevel] = useState<number>(0);

  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const silentOutputRef = useRef<GainNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const onAudioChunkRef = useRef(onAudioChunk);

  useEffect(() => {
    onAudioChunkRef.current = onAudioChunk;
  }, [onAudioChunk]);

  const startRecording = useCallback(async () => {
    setMicError(null);

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setMicError('Your browser does not support microphone audio capture.');
      return false;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      streamRef.current = stream;

      // Initialize Web Audio API
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      const audioCtx = new AudioCtx({ sampleRate });
      if (audioCtx.sampleRate !== sampleRate) {
        await audioCtx.close();
        stream.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        setMicError(`This browser provides ${audioCtx.sampleRate} Hz audio; ${sampleRate} Hz is required.`);
        return false;
      }
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      sourceRef.current = source;

      // 1024 or 2048 buffer length
      const bufferSize = 2048;
      const processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);
      processorRef.current = processor;
      const silentOutput = audioCtx.createGain();
      silentOutput.gain.value = 0;
      silentOutputRef.current = silentOutput;

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        
        // Calculate instantaneous RMS level for UI visualizer
        let sum = 0;
        for (let i = 0; i < inputData.length; i++) {
          sum += inputData[i] * inputData[i];
        }
        const rms = Math.sqrt(sum / inputData.length);
        setAudioLevel(Math.min(1.0, rms * 4.0));

        // Resample/convert to 16-bit PCM (little-endian)
        const pcm16 = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }

        if (onAudioChunkRef.current) {
          onAudioChunkRef.current(pcm16.buffer);
        }
      };

      source.connect(processor);
      processor.connect(silentOutput);
      silentOutput.connect(audioCtx.destination);

      setIsRecording(true);
      return true;
    } catch (err: any) {
      console.error('Microphone capture error:', err);
      processorRef.current?.disconnect();
      sourceRef.current?.disconnect();
      silentOutputRef.current?.disconnect();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        void audioContextRef.current.close();
      }
      audioContextRef.current = null;
      processorRef.current = null;
      sourceRef.current = null;
      silentOutputRef.current = null;
      let message = 'Failed to access microphone.';
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        message = 'Microphone permission was denied. Please allow microphone access in your browser address bar.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        message = 'No microphone device found on your system. You can test the application using Demo Mode.';
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        message = 'Microphone is already occupied by another application.';
      }
      setMicError(message);
      setIsRecording(false);
      return false;
    }
  }, [sampleRate]);

  const stopRecording = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    if (silentOutputRef.current) {
      silentOutputRef.current.disconnect();
      silentOutputRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    setIsRecording(false);
    setAudioLevel(0);
  }, []);

  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, [stopRecording]);

  return {
    isRecording,
    micError,
    audioLevel,
    startRecording,
    stopRecording,
    clearError: () => setMicError(null)
  };
}
