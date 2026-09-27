import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useAudioCapture } from '../hooks/useAudioCapture';

const originalAudioContext = Object.getOwnPropertyDescriptor(window, 'AudioContext');
const originalMediaDevices = Object.getOwnPropertyDescriptor(navigator, 'mediaDevices');

afterEach(() => {
  if (originalAudioContext) {
    Object.defineProperty(window, 'AudioContext', originalAudioContext);
  } else {
    Reflect.deleteProperty(window, 'AudioContext');
  }
  if (originalMediaDevices) {
    Object.defineProperty(navigator, 'mediaDevices', originalMediaDevices);
  } else {
    Reflect.deleteProperty(navigator, 'mediaDevices');
  }
});

function installAudioMocks(actualSampleRate = 16000) {
  const track = { stop: vi.fn() };
  const stream = { getTracks: () => [track] } as unknown as MediaStream;
  const source = { connect: vi.fn(), disconnect: vi.fn() };
  const processor: {
    connect: ReturnType<typeof vi.fn>;
    disconnect: ReturnType<typeof vi.fn>;
    onaudioprocess: ((event: AudioProcessingEvent) => void) | null;
  } = { connect: vi.fn(), disconnect: vi.fn(), onaudioprocess: null };
  const silentOutput = {
    gain: { value: 1 },
    connect: vi.fn(),
    disconnect: vi.fn()
  };
  const context = {
    sampleRate: actualSampleRate,
    state: 'running',
    destination: {},
    createMediaStreamSource: vi.fn(() => source),
    createScriptProcessor: vi.fn(() => processor),
    createGain: vi.fn(() => silentOutput),
    close: vi.fn().mockResolvedValue(undefined)
  };
  const getUserMedia = vi.fn().mockResolvedValue(stream);

  Object.defineProperty(window, 'AudioContext', {
    configurable: true,
    value: function MockAudioContext() {
      return context;
    } as unknown as typeof AudioContext
  });
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia }
  });

  return { context, getUserMedia, processor, silentOutput, source, stream, track };
}

describe('useAudioCapture', () => {
  it('streams PCM16 at 16 kHz through a silent output and releases the mic on cleanup', async () => {
    const mocks = installAudioMocks();
    const onAudioChunk = vi.fn();
    const { result, unmount } = renderHook(() => useAudioCapture({ onAudioChunk }));

    let started = false;
    await act(async () => {
      started = await result.current.startRecording();
    });

    expect(started).toBe(true);
    expect(mocks.getUserMedia).toHaveBeenCalledOnce();
    expect(mocks.context.createGain).toHaveBeenCalledOnce();
    expect(mocks.silentOutput.gain.value).toBe(0);
    expect(mocks.processor.connect).toHaveBeenCalledWith(mocks.silentOutput);
    expect(mocks.silentOutput.connect).toHaveBeenCalledWith(mocks.context.destination);

    const audioEvent = {
      inputBuffer: { getChannelData: () => new Float32Array([-1, 0, 1]) }
    } as unknown as AudioProcessingEvent;
    await act(async () => {
      mocks.processor.onaudioprocess?.(audioEvent);
    });
    expect(Array.from(new Int16Array(onAudioChunk.mock.calls[0][0]))).toEqual([-32768, 0, 32767]);

    unmount();
    expect(mocks.track.stop).toHaveBeenCalledOnce();
    expect(mocks.context.close).toHaveBeenCalledOnce();
  });

  it('rejects a browser audio context with the wrong sample rate', async () => {
    const mocks = installAudioMocks(44100);
    const { result } = renderHook(() => useAudioCapture());

    let started = true;
    await act(async () => {
      started = await result.current.startRecording();
    });

    expect(started).toBe(false);
    expect(result.current.micError).toContain('44100 Hz');
    expect(mocks.track.stop).toHaveBeenCalledOnce();
    expect(mocks.context.close).toHaveBeenCalledOnce();
    expect(mocks.context.createMediaStreamSource).not.toHaveBeenCalled();
  });
});
