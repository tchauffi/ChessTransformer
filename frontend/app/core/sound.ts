// Board sounds, synthesised with the Web Audio API so the demo ships no audio
// assets: a filtered-noise "knock" for wood on wood, plus short tones for
// check, game start/end and notices.

export type SoundName =
  | 'move'
  | 'opponent'
  | 'capture'
  | 'castle'
  | 'check'
  | 'promote'
  | 'premove'
  | 'illegal'
  | 'start'
  | 'end'
  | 'notify'
  | 'lowtime';

let ctx: AudioContext | null = null;
let master: GainNode | null = null;
let noise: AudioBuffer | null = null;

function audio(): AudioContext | null {
  if (typeof window === 'undefined') return null;
  if (!ctx) {
    const AC =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AC) return null;
    ctx = new AC();
    master = ctx.createGain();
    master.gain.value = 0.7;
    master.connect(ctx.destination);
  }
  if (ctx.state === 'suspended') void ctx.resume();
  return ctx;
}

function noiseBuffer(c: AudioContext): AudioBuffer {
  if (!noise) {
    noise = c.createBuffer(1, Math.floor(c.sampleRate * 0.25), c.sampleRate);
    const data = noise.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
  }
  return noise;
}

function envelope(c: AudioContext, t: number, peak: number, decay: number): GainNode {
  const g = c.createGain();
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(peak, t + 0.004);
  g.gain.exponentialRampToValueAtTime(0.0001, t + decay);
  g.connect(master!);
  return g;
}

/** A piece set down on the board: a bandpassed click with a short low body. */
function knock(c: AudioContext, t: number, { freq = 1500, body = 190, gain = 0.55, decay = 0.07 } = {}) {
  const src = c.createBufferSource();
  src.buffer = noiseBuffer(c);
  const bp = c.createBiquadFilter();
  bp.type = 'bandpass';
  bp.frequency.value = freq;
  bp.Q.value = 1.6;
  src.connect(bp).connect(envelope(c, t, gain, decay));
  src.start(t);
  src.stop(t + decay + 0.02);

  const osc = c.createOscillator();
  osc.frequency.setValueAtTime(body, t);
  osc.frequency.exponentialRampToValueAtTime(body * 0.55, t + 0.09);
  osc.connect(envelope(c, t, gain * 0.7, 0.1));
  osc.start(t);
  osc.stop(t + 0.12);
}

function tone(c: AudioContext, t: number, freq: number, dur: number, gain = 0.16, type: OscillatorType = 'sine') {
  const osc = c.createOscillator();
  osc.type = type;
  osc.frequency.value = freq;
  osc.connect(envelope(c, t, gain, dur));
  osc.start(t);
  osc.stop(t + dur + 0.02);
}

export function playSound(name: SoundName) {
  const c = audio();
  if (!c) return;
  const t = c.currentTime + 0.01;
  switch (name) {
    case 'move':
      knock(c, t);
      break;
    case 'opponent':
      knock(c, t, { freq: 1150, body: 160 });
      break;
    case 'capture':
      knock(c, t, { freq: 2300, body: 230, gain: 0.7, decay: 0.05 });
      knock(c, t + 0.03, { freq: 1300, body: 150, gain: 0.5 });
      break;
    case 'castle':
      knock(c, t);
      knock(c, t + 0.1, { freq: 1250, body: 170, gain: 0.45 });
      break;
    case 'check':
      knock(c, t);
      tone(c, t + 0.02, 1046, 0.18, 0.09, 'triangle');
      tone(c, t + 0.02, 1568, 0.14, 0.05, 'triangle');
      break;
    case 'promote':
      knock(c, t);
      tone(c, t + 0.03, 784, 0.12, 0.08);
      tone(c, t + 0.1, 1175, 0.18, 0.08);
      break;
    case 'premove':
      knock(c, t, { freq: 2600, body: 260, gain: 0.25, decay: 0.04 });
      break;
    case 'illegal':
      tone(c, t, 150, 0.14, 0.12, 'square');
      break;
    case 'start':
      tone(c, t, 523, 0.22, 0.13);
      tone(c, t + 0.11, 784, 0.32, 0.13);
      break;
    case 'end':
      tone(c, t, 784, 0.2, 0.13);
      tone(c, t + 0.12, 659, 0.2, 0.13);
      tone(c, t + 0.24, 523, 0.45, 0.14);
      break;
    case 'notify':
      tone(c, t, 880, 0.2, 0.1);
      break;
    case 'lowtime':
      tone(c, t, 1318, 0.08, 0.1, 'triangle');
      tone(c, t + 0.14, 1318, 0.08, 0.1, 'triangle');
      break;
  }
}
