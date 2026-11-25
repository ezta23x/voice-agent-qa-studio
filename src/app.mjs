import { chromium } from 'playwright';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { execFileSync, spawnSync } from 'child_process';
import { fileURLToPath, pathToFileURL } from 'url';
import {
  getSettings, updateSettings, rollRandomScenarioContext,
  synthesizeSpeech, streamSynthesizeSpeech, decideNextCallerTurn
} from './gemini.mjs';
import {
  ISSUE_TAG_PRESETS, ISSUE_TAG_SPECS, inferSuspectedTechIssues,
  buildObjectiveTurnEval, buildRunReport, saveHumanReview, saveRunNote
} from './evaluator.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');
const ENV_PATH = path.join(ROOT_DIR, '.env');
const CFG_PATH = path.join(ROOT_DIR, 'config', 'scenarios.json');
const RESULTS_ROOT = path.join(ROOT_DIR, 'results');
const FFMPEG_BIN = ['/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg'].find(p => fs.existsSync(p)) || 'ffmpeg';
fs.mkdirSync(RESULTS_ROOT, { recursive: true });

// 内置安全解析 .env（防止 URL 中的 & / ? 等符号被外部 Shell source 截断）
function loadDotEnvSafe() {
  if (!fs.existsSync(ENV_PATH)) return;
  for (const rawLine of fs.readFileSync(ENV_PATH, 'utf8').split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const eq = line.indexOf('=');
    if (eq <= 0) continue;
    const k = line.slice(0, eq).trim();
    let v = line.slice(eq + 1).trim();
    if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) v = v.slice(1, -1);
    if (!process.env[k]) process.env[k] = v;
  }
}
loadDotEnvSafe();
updateSettings({
  provider: process.env.API_PROVIDER,
  vertexApiKey: process.env.VERTEX_API_KEY,
  geminiApiKey: process.env.GEMINI_API_KEY,
  callerModel: process.env.VERTEX_MODEL,
  ttsModel: process.env.VERTEX_TTS_MODEL,
  ttsVoice: process.env.VERTEX_TTS_VOICE,
  randomizeScenario: process.env.RANDOMIZE_SCENARIO !== 'false'
});

let cfg = JSON.parse(fs.readFileSync(CFG_PATH, 'utf8'));
let targetUrl = process.env.TARGET_URL || 'https://example.com';

let uiPage = null;
let targetPage = null;
let cdpSession = null;
let isRunning = false;
let abortRequested = false;
let runAbortController = null;
let humanTakeoverNextTurn = false;
let humanDoneResolver = null;
let lastScreenDraft = null; // 暂存最近一场待手动导出的录屏帧，不自动写入 results/

function persistEnvSettings(st, url) {
  const lines = [
    `TARGET_URL=${url}`,
    `API_PROVIDER=${st.provider}`,
    `VERTEX_API_KEY=${st.vertexApiKey}`,
    `GEMINI_API_KEY=${st.geminiApiKey}`,
    `VERTEX_MODEL=${st.callerModel}`,
    `VERTEX_TTS_MODEL=${st.ttsModel}`,
    `VERTEX_TTS_VOICE=${st.ttsVoice}`,
    `RANDOMIZE_SCENARIO=${st.randomizeScenario ? 'true' : 'false'}`
  ];
  fs.writeFileSync(ENV_PATH, lines.join('\n') + '\n', { mode: 0o600 });
  try { fs.chmodSync(ENV_PATH, 0o600); } catch {}
}

function emitUi(type, payload) {
  if (!uiPage || uiPage.isClosed()) return;
  uiPage.evaluate(([t, p]) => window.__onAppEvent && window.__onAppEvent(t, p), [type, payload]).catch(() => {});
}

function log(msg, level = 'info') {
  console.log(`[${level.toUpperCase()}] ${msg}`);
  emitUi('log', { ts: new Date().toLocaleTimeString(), level, msg });
}

// Browser-side WebRTC audio router, virtual mic + optional 1-turn physical mic passthrough,
// native inbound-rtp VAD, per-turn cloned MediaRecorder, and passive WebSocket/DataChannel ASR tap.
const INIT_SCRIPT = () => {
  const state = {
    micCtx: null, micKeepAliveGain: null, activeMicDests: new Set(),
    origGUM: null, humanMicStream: null, humanMicSource: null, humanMicGain: null, humanMicActive: false,
    pcs: new Set(),
    recorder: null, chunks: [],
    turnChunks: [], turnRecorder: null, turnClonedTracks: [], wantTurnRecording: false,
    attachedTracks: new Set(), remoteTracks: [], capturedMediaEls: new WeakSet(),
    aiSpeaking: false, aiStartTs: 0, aiLastVoiceTs: 0, lastAudioLevel: 0, peakAudioLevel: 0, lastEnergyMap: new Map(),
    netAsrEvents: []
  };
  window.__vaTest = state;

  function ensureMicCtx() {
    if (!state.micCtx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      state.micCtx = new AC({ sampleRate: 48000 });
      const osc = state.micCtx.createOscillator();
      const gain = state.micCtx.createGain();
      gain.gain.value = 0.00001;
      osc.connect(gain);
      osc.start();
      state.micKeepAliveGain = gain;
    }
    if (state.micCtx.state === 'suspended') state.micCtx.resume().catch(() => {});
  }

  function createFreshMicStream() {
    ensureMicCtx();
    for (const d of [...state.activeMicDests]) {
      if (d.stream.getAudioTracks().every(t => t.readyState !== 'live')) state.activeMicDests.delete(d);
    }
    const dest = state.micCtx.createMediaStreamDestination();
    state.micKeepAliveGain.connect(dest);
    if (state.humanMicGain) {
      try { state.humanMicGain.connect(dest); } catch {}
    }
    state.activeMicDests.add(dest);
    return dest.stream;
  }

  // Toggle real physical microphone passthrough ONLY during 1-turn human takeover
  window.__setHumanMicActive = async (on) => {
    ensureMicCtx();
    if (state.micCtx.state === 'suspended') await state.micCtx.resume();
    state.humanMicActive = Boolean(on);
    if (on) {
      if (!state.humanMicSource && state.origGUM) {
        try {
          state.humanMicStream = await state.origGUM({ audio: true });
          state.humanMicSource = state.micCtx.createMediaStreamSource(state.humanMicStream);
          state.humanMicGain = state.micCtx.createGain();
          state.humanMicGain.gain.value = 1.0;
          state.humanMicSource.connect(state.humanMicGain);
          for (const dest of state.activeMicDests) {
            if (dest.stream.getAudioTracks().some(t => t.readyState === 'live')) {
              try { state.humanMicGain.connect(dest); } catch {}
            }
          }
        } catch {
          return false;
        }
      } else if (state.humanMicGain) {
        state.humanMicGain.gain.value = 1.0;
        for (const dest of state.activeMicDests) {
          if (dest.stream.getAudioTracks().some(t => t.readyState === 'live')) {
            try { state.humanMicGain.connect(dest); } catch {}
          }
        }
      }
      return true;
    } else {
      if (state.humanMicGain) state.humanMicGain.gain.value = 0.0;
      return true;
    }
  };

  // Passive network transcript tap (WebSocket + RTCDataChannel) as backup for ASR capture
  function tapAsrMessage(raw) {
    if (typeof raw !== 'string' || raw.length < 4 || raw.length > 20000) return;
    if (!/asr|transcript|user|text|recognize|query|content/i.test(raw)) return;
    try {
      const obj = JSON.parse(raw);
      const role = String(obj.role || obj.speaker || obj.type || obj.event || '').toLowerCase();
      const txt = obj.asr_text || obj.transcript || obj.query || (role.includes('user') || role.includes('asr') ? (obj.text || obj.content || obj.data?.text) : '');
      if (typeof txt === 'string' && txt.trim()) {
        state.netAsrEvents.push({ ts: Date.now(), text: txt.trim() });
        if (state.netAsrEvents.length > 100) state.netAsrEvents.shift();
      }
    } catch {}
  }

  // Lock all popups / target="_blank" links inside the embedded target tab
  window.open = function(url) {
    if (url && typeof url === 'string') {
      try { window.location.href = new URL(url, window.location.href).href; } catch { window.location.href = url; }
    }
    return window;
  };
  document.addEventListener('click', (e) => {
    const a = e.target && e.target.closest ? e.target.closest('a[target="_blank"]') : null;
    if (a) a.setAttribute('target', '_self');
  }, true);

  const OrigWS = window.WebSocket;
  if (OrigWS) {
    window.WebSocket = function(...args) {
      const ws = new OrigWS(...args);
      ws.addEventListener('message', (ev) => tapAsrMessage(ev.data));
      return ws;
    };
    window.WebSocket.prototype = OrigWS.prototype;
    Object.assign(window.WebSocket, OrigWS);
  }

  // Native WebRTC inbound-rtp audioLevel + totalAudioEnergy monitor (runs every 80ms)
  setInterval(async () => {
    let maxLevel = 0;
    let energyActive = false;
    for (const pc of state.pcs) {
      if (pc.connectionState === 'closed') { state.pcs.delete(pc); continue; }
      try {
        const stats = await pc.getStats();
        stats.forEach((r) => {
          if (r.type === 'inbound-rtp' && (r.kind === 'audio' || r.mediaType === 'audio')) {
            if (typeof r.audioLevel === 'number' && r.audioLevel > maxLevel) maxLevel = r.audioLevel;
            if (typeof r.totalAudioEnergy === 'number') {
              const prev = state.lastEnergyMap.get(r.id) ?? r.totalAudioEnergy;
              if (r.totalAudioEnergy - prev > 0.0002) energyActive = true;
              state.lastEnergyMap.set(r.id, r.totalAudioEnergy);
            }
          }
        });
      } catch {}
    }
    state.lastAudioLevel = maxLevel;
    if (maxLevel > state.peakAudioLevel) state.peakAudioLevel = maxLevel;
    const now = Date.now();
    if (maxLevel > 0.01 || energyActive) {
      if (!state.aiSpeaking) { state.aiSpeaking = true; state.aiStartTs = now; }
      state.aiLastVoiceTs = now;
    } else if (state.aiSpeaking && (now - state.aiLastVoiceTs > 900)) {
      state.aiSpeaking = false;
    }
  }, 80);

  function getLiveRemoteTracks() {
    const active = state.remoteTracks.filter(t => t.readyState === 'live' && !t.muted);
    return active.length ? active : state.remoteTracks.filter(t => t.readyState === 'live');
  }

  function startTurnRecorderNow() {
    const live = getLiveRemoteTracks();
    if (!live.length) return false;
    if (state.turnRecorder && state.turnRecorder.state === 'recording') return true;
    state.turnChunks = [];
    state.turnClonedTracks.forEach(t => { try { t.stop(); } catch {} });
    state.turnClonedTracks = live.map(t => { try { return t.clone(); } catch { return t; } });
    const stream = new MediaStream(state.turnClonedTracks);
    try {
      const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm';
      state.turnRecorder = new MediaRecorder(stream, { mimeType: mime });
      state.turnRecorder.ondataavailable = (e) => { if (e.data && e.data.size > 0) state.turnChunks.push(e.data); };
      state.turnRecorder.start(150);
      return true;
    } catch { return false; }
  }

  function ensureFullRecorder() {
    if (state.recorder && state.recorder.state === 'recording') return;
    const live = getLiveRemoteTracks();
    if (!live.length) return;
    try {
      const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm';
      state.recorder = new MediaRecorder(new MediaStream(live), { mimeType: mime });
      state.recorder.ondataavailable = (e) => { if (e.data && e.data.size > 0) state.chunks.push(e.data); };
      state.recorder.start(250);
    } catch {}
  }

  function attachRemoteStream(stream) {
    if (!stream || !stream.getAudioTracks) return;
    for (const track of stream.getAudioTracks()) {
      if (state.attachedTracks.has(track.id)) continue;
      state.attachedTracks.add(track.id);
      state.remoteTracks.push(track);

      const onTrackReady = () => {
        ensureFullRecorder();
        if (state.wantTurnRecording && (!state.turnRecorder || state.turnRecorder.state !== 'recording')) {
          startTurnRecorderNow();
        }
      };
      onTrackReady();
      track.addEventListener('unmute', onTrackReady);
    }
  }

  function inspectMediaElement(el) {
    if (!el) return;
    try {
      if (el.srcObject && el.srcObject.getAudioTracks && el.srcObject.getAudioTracks().length) {
        attachRemoteStream(el.srcObject);
      } else if (!state.capturedMediaEls.has(el) && typeof el.captureStream === 'function') {
        const cap = el.captureStream();
        if (cap && cap.getAudioTracks && cap.getAudioTracks().length) {
          state.capturedMediaEls.add(el);
          attachRemoteStream(cap);
        }
      }
    } catch {}
  }

  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    state.origGUM = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async (c) => (c && c.audio) ? createFreshMicStream() : state.origGUM(c);
  }

  const OrigPC = window.RTCPeerConnection;
  if (OrigPC) {
    window.RTCPeerConnection = function(...args) {
      const pc = new OrigPC(...args);
      state.pcs.add(pc);
      pc.addEventListener('track', (ev) => {
        if (ev.track && ev.track.kind === 'audio') attachRemoteStream(ev.streams[0] || new MediaStream([ev.track]));
      });
      pc.addEventListener('datachannel', (ev) => {
        if (ev.channel) ev.channel.addEventListener('message', (m) => tapAsrMessage(m.data));
      });
      return pc;
    };
    window.RTCPeerConnection.prototype = OrigPC.prototype;
  }

  const desc = Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, 'srcObject');
  if (desc && desc.set) {
    Object.defineProperty(HTMLMediaElement.prototype, 'srcObject', {
      get: desc.get,
      set(val) {
        const res = desc.set.call(this, val);
        inspectMediaElement(this);
        return res;
      }
    });
  }
  const origPlay = HTMLMediaElement.prototype.play;
  if (origPlay) {
    HTMLMediaElement.prototype.play = function(...args) {
      inspectMediaElement(this);
      const p = origPlay.apply(this, args);
      if (p && typeof p.then === 'function') p.then(() => inspectMediaElement(this)).catch(() => {});
      return p;
    };
  }

  window.__playUserWavBase64 = async (b64) => {
    ensureMicCtx();
    if (state.micCtx.state === 'suspended') await state.micCtx.resume();
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    const audioBuf = await state.micCtx.decodeAudioData(arr.buffer);
    const src = state.micCtx.createBufferSource();
    src.buffer = audioBuf;
    let liveCount = 0;
    for (const dest of [...state.activeMicDests]) {
      const hasLive = dest.stream.getAudioTracks().some(t => t.readyState === 'live');
      if (hasLive) { src.connect(dest); liveCount++; }
      else state.activeMicDests.delete(dest);
    }
    const monitorGain = state.micCtx.createGain();
    monitorGain.gain.value = 0.85;
    src.connect(monitorGain);
    monitorGain.connect(state.micCtx.destination);
    const startTs = Date.now();
    await new Promise(resolve => {
      state.activeWavSrc = src;
      state.activeWavResolve = resolve;
      src.onended = () => { state.activeWavSrc = null; state.activeWavResolve = null; resolve(); };
      src.start();
    });
    return { startTs, endTs: Date.now(), durationMs: Math.round(audioBuf.duration * 1000), liveMicTracks: liveCount };
  };

  window.__beginUserAudioStream = async (telephonyFilter = false) => {
    ensureMicCtx();
    if (state.micCtx.state === 'suspended') await state.micCtx.resume();
    state.streamAbort = false;
    const bus = state.micCtx.createGain();
    bus.gain.value = 1.0;
    let outNode = bus;
    if (telephonyFilter) {
      const hp = state.micCtx.createBiquadFilter(); hp.type = 'highpass'; hp.frequency.value = 300;
      const lp = state.micCtx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 3400;
      bus.connect(hp); hp.connect(lp); outNode = lp;
    }
    let liveCount = 0;
    for (const dest of [...state.activeMicDests]) {
      const hasLive = dest.stream.getAudioTracks().some(t => t.readyState === 'live');
      if (hasLive) { outNode.connect(dest); liveCount++; }
      else state.activeMicDests.delete(dest);
    }
    const mon = state.micCtx.createGain();
    mon.gain.value = 0.85;
    outNode.connect(mon);
    mon.connect(state.micCtx.destination);
    state.streamBus = bus;
    state.streamOutNode = outNode;
    state.streamNextTime = 0;
    state.streamStartTs = 0;
    state.streamTotalDur = 0;
    state.streamLiveCount = liveCount;
    return liveCount;
  };

  window.__pushUserPcmChunk = (b64, sampleRate = 24000) => {
    if (!state.micCtx || !state.streamBus) return;
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const samples = Math.floor(bytes.length / 2);
    if (samples <= 0) return;
    const dv = new DataView(bytes.buffer, bytes.byteOffset, samples * 2);
    const audioBuf = state.micCtx.createBuffer(1, samples, sampleRate);
    const ch = audioBuf.getChannelData(0);
    for (let i = 0; i < samples; i++) ch[i] = dv.getInt16(i * 2, true) / 32768;
    const src = state.micCtx.createBufferSource();
    src.buffer = audioBuf;
    src.connect(state.streamBus);
    const now = state.micCtx.currentTime;
    const startAt = state.streamNextTime > now ? state.streamNextTime : now + 0.02;
    src.start(startAt);
    if (!state.streamStartTs) state.streamStartTs = Date.now();
    state.streamNextTime = startAt + audioBuf.duration;
    state.streamTotalDur += audioBuf.duration;
  };

  window.__stopUserAudioNow = () => {
    state.streamAbort = true;
    try { if (state.activeWavSrc) state.activeWavSrc.stop(); } catch {}
    try { if (state.activeWavResolve) state.activeWavResolve(); } catch {}
    state.activeWavSrc = null; state.activeWavResolve = null;
    try { if (state.streamBus) state.streamBus.disconnect(); } catch {}
    try { if (state.streamOutNode) state.streamOutNode.disconnect(); } catch {}
    state.streamBus = null; state.streamOutNode = null; state.streamNextTime = 0;
  };

  window.__finishUserAudioStream = async () => {
    if (state.micCtx && state.streamNextTime > state.micCtx.currentTime) {
      const remainMs = Math.ceil((state.streamNextTime - state.micCtx.currentTime) * 1000) + 40;
      const endAt = Date.now() + Math.min(remainMs, 30000);
      while (Date.now() < endAt && !state.streamAbort) {
        await new Promise(r => setTimeout(r, 60));
      }
    }
    state.streamAbort = false;
    const endTs = Date.now();
    const res = {
      startTs: state.streamStartTs || endTs,
      endTs,
      durationMs: Math.round((state.streamTotalDur || 0) * 1000),
      liveMicTracks: state.streamLiveCount || 0
    };
    state.streamBus = null; state.streamOutNode = null;
    return res;
  };

  window.__startTurnRecording = () => {
    document.querySelectorAll('audio, video').forEach(inspectMediaElement);
    state.wantTurnRecording = true;
    state.turnChunks = [];
    return startTurnRecorderNow();
  };

  window.__stopTurnRecordingBase64 = async () => {
    state.wantTurnRecording = false;
    if (!state.turnRecorder || state.turnRecorder.state === 'inactive') return '';
    await new Promise(resolve => { state.turnRecorder.onstop = resolve; state.turnRecorder.stop(); });
    state.turnRecorder = null;
    state.turnClonedTracks.forEach(t => { try { t.stop(); } catch {} });
    state.turnClonedTracks = [];
    const blob = new Blob(state.turnChunks, { type: 'audio/webm' });
    const bytes = new Uint8Array(await blob.arrayBuffer());
    let bin = '';
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    return btoa(bin);
  };

  window.__stopFullRecordingBase64 = async () => {
    if (!state.recorder || state.recorder.state === 'inactive') { state.chunks = []; return ''; }
    await new Promise(resolve => { state.recorder.onstop = resolve; state.recorder.stop(); });
    state.recorder = null;
    const blob = new Blob(state.chunks, { type: 'audio/webm' });
    state.chunks = [];
    state.remoteTracks = state.remoteTracks.filter(t => t.readyState === 'live');
    const bytes = new Uint8Array(await blob.arrayBuffer());
    let bin = '';
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    return btoa(bin);
  };
};

// 清理上一次未手动导出的临时录屏帧目录，防止占用磁盘
function clearLastScreenDraft() {
  if (lastScreenDraft?.framesDir) {
    try { fs.rmSync(lastScreenDraft.framesDir, { recursive: true, force: true }); } catch {}
  }
  lastScreenDraft = null;
}

// 按需 CDP 录屏器：通话期间仅缓存在系统临时目录，用户点击顶栏「保存完整录屏」后再合成 MP4 写入 results/
function createScreenRecorder() {
  clearLastScreenDraft();
  const framesDir = path.join(os.tmpdir(), `va_qa_frames_${Date.now()}`);
  fs.mkdirSync(framesDir, { recursive: true });
  const frames = [];
  let active = false;
  let handler = null;

  return {
    async start() {
      if (!cdpSession) return;
      active = true;
      handler = async (ev) => {
        if (!active) return;
        try {
          await cdpSession.send('Page.screencastFrameAck', { sessionId: ev.sessionId }).catch(() => {});
          const idx = frames.length;
          const file = path.join(framesDir, `f_${String(idx).padStart(5, '0')}.jpg`);
          fs.writeFileSync(file, Buffer.from(ev.data, 'base64'));
          frames.push({ file, ts: Date.now() });
        } catch {}
      };
      cdpSession.on('Page.screencastFrame', handler);
      await cdpSession.send('Page.startScreencast', { format: 'jpeg', quality: 68, everyNthFrame: 2 }).catch(() => {});
    },
    async stopCapture() {
      active = false;
      if (cdpSession) {
        await cdpSession.send('Page.stopScreencast').catch(() => {});
        if (handler) cdpSession.off('Page.screencastFrame', handler);
      }
      return { framesDir, frames };
    }
  };
}

// 将暂存帧与整场音频合成为 MP4 并写入 results/<runId>/<scenarioId>/session_recording.mp4
function muxDraftToMp4(draft) {
  if (!draft || !draft.frames || draft.frames.length < 2) return null;
  const mp4Path = path.join(draft.scOutDir, 'session_recording.mp4');
  try {
    const listPath = path.join(draft.framesDir, 'frames.txt');
    const lines = [];
    for (let i = 0; i < draft.frames.length; i++) {
      const durSec = i < draft.frames.length - 1
        ? Math.max(0.05, Math.min(5.0, (draft.frames[i + 1].ts - draft.frames[i].ts) / 1000))
        : 0.5;
      lines.push(`file '${draft.frames[i].file.replace(/'/g, "'\\''")}'`);
      lines.push(`duration ${durSec.toFixed(3)}`);
    }
    lines.push(`file '${draft.frames[draft.frames.length - 1].file.replace(/'/g, "'\\''")}'`);
    fs.writeFileSync(listPath, lines.join('\n'), 'utf8');

    const args = ['-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', listPath];
    const hasAudio = draft.audioWavPath && fs.existsSync(draft.audioWavPath) && fs.statSync(draft.audioWavPath).size > 2000;
    if (hasAudio) args.push('-i', draft.audioWavPath);
    args.push('-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-r', '12');
    if (hasAudio) args.push('-c:a', 'aac', '-b:a', '128k', '-shortest');
    args.push(mp4Path);

    const res = spawnSync(FFMPEG_BIN, args, { encoding: 'utf8' });
    if (res.status === 0 && fs.existsSync(mp4Path)) return mp4Path;
  } catch {}
  return null;
}

function saveWebmAndWav(b64, outDir, baseName) {
  if (!b64) return null;
  const webmPath = path.join(outDir, `${baseName}.webm`);
  const wavPath = path.join(outDir, `${baseName}.wav`);
  fs.writeFileSync(webmPath, Buffer.from(b64, 'base64'));
  try {
    execFileSync(FFMPEG_BIN, ['-y', '-loglevel', 'error', '-i', webmPath, '-ar', '24000', '-ac', '1', wavPath]);
    return wavPath;
  } catch {
    return webmPath;
  }
}

function toFileSrc(absPath) {
  return absPath && fs.existsSync(absPath) ? pathToFileURL(absPath).href : '';
}

async function getLatestDelta(prevText) {
  const fullText = await targetPage.evaluate(() => document.body.innerText || '');
  if (!prevText) return { fullText, delta: fullText.slice(-800) };
  const prevLines = new Set(prevText.split('\n').map(s => s.trim()).filter(Boolean));
  const newLines = fullText.split('\n').map(s => s.trim()).filter(s => s && !prevLines.has(s));
  return { fullText, delta: newLines.join('\n') };
}

const UI_CHROME_LINES = new Set([
  '实时连接', '麦克风已开启', '用自然的语气与 Agent 对话', '已连接',
  '准备就绪', '连接后即可开始语音测试', '开始对话', '结束对话', '对话已结束', '音频播放', '返回评测工作台'
]);

// Split page delta cleanly into exact customer-service ASR transcript (「你」) vs Agent reply (「Agent」)
function splitTurnDelta(delta) {
  const lines = String(delta || '').split('\n').map(s => s.trim()).filter(Boolean);
  const userLines = [];
  const agentLines = [];
  const nodeEvents = [];
  let speaker = 'agent';
  for (const line of lines) {
    if (line === '你' || line === 'User') { speaker = 'user'; continue; }
    if (line === 'Agent' || /^AI/i.test(line)) { speaker = 'agent'; continue; }
    if (line.startsWith('进入节点')) { nodeEvents.push(line); continue; }
    if (/^\d{2}:\d{2}:\d{2}$/.test(line) || line === '节点切换' || UI_CHROME_LINES.has(line)) continue;
    if (speaker === 'user') userLines.push(line);
    else agentLines.push(line);
  }
  return {
    userAsrText: userLines.join('\n').trim(),
    agentText: agentLines.join('\n').trim(),
    nodeEvents
  };
}

async function getNetAsrSince(sinceTs) {
  return await targetPage.evaluate((ts) => {
    const arr = (window.__vaTest?.netAsrEvents || []).filter(e => e.ts >= ts);
    return arr.length ? arr[arr.length - 1].text : '';
  }, sinceTs).catch(() => '');
}

async function waitForAiTurn(sinceTs, prevText, { maxWaitStartMs = 15000, silenceAfterMs = 950, maxTurnMs = 45000, stopAfterStartMs = 0 } = {}) {
  let detectedStart = 0;
  let usedAudio = false;
  let lastDomChangeTs = 0;
  let lastDomDelta = '';
  const deadline = Date.now() + maxWaitStartMs;

  while (Date.now() < deadline && !abortRequested) {
    const st = await targetPage.evaluate(() => ({
      pcs: window.__vaTest?.pcs?.size || 0,
      startTs: window.__vaTest?.aiStartTs || 0,
      lastVoiceTs: window.__vaTest?.aiLastVoiceTs || 0
    }));
    if (st.lastVoiceTs > sinceTs) {
      detectedStart = Math.max(sinceTs, st.startTs > sinceTs ? st.startTs : st.lastVoiceTs);
      usedAudio = true;
      break;
    }
    if (st.pcs === 0) {
      const dom = await getLatestDelta(prevText);
      const { agentText } = splitTurnDelta(dom.delta);
      if (/[\u3040-\u30ff]/.test(agentText) && agentText !== lastDomDelta) {
        lastDomDelta = agentText;
        lastDomChangeTs = Date.now();
        detectedStart = lastDomChangeTs;
        break;
      }
    }
    await targetPage.waitForTimeout(70);
  }

  if (!detectedStart) return { responded: false, ttfbMs: null, durationMs: 0 };
  if (stopAfterStartMs > 0) {
    await targetPage.waitForTimeout(stopAfterStartMs);
    return { responded: true, ttfbMs: detectedStart - sinceTs, durationMs: stopAfterStartMs, interrupted: true };
  }

  const endDeadline = Date.now() + maxTurnMs;
  while (Date.now() < endDeadline && !abortRequested) {
    const st = await targetPage.evaluate(() => ({ lastVoiceTs: window.__vaTest?.aiLastVoiceTs || 0 }));
    let lastActivity = st.lastVoiceTs > sinceTs ? st.lastVoiceTs : 0;
    if (!usedAudio) {
      const dom = await getLatestDelta(prevText);
      const { agentText } = splitTurnDelta(dom.delta);
      if (agentText && agentText !== lastDomDelta) {
        lastDomDelta = agentText;
        lastDomChangeTs = Date.now();
      }
      lastActivity = Math.max(lastActivity, lastDomChangeTs);
    }
    const now = Date.now();
    if (lastActivity > 0 && (now - detectedStart >= 400) && (now - lastActivity >= silenceAfterMs)) {
      return { responded: true, ttfbMs: detectedStart - sinceTs, durationMs: lastActivity - detectedStart };
    }
    await targetPage.waitForTimeout(80);
  }
  return { responded: true, ttfbMs: detectedStart - sinceTs, durationMs: Date.now() - detectedStart, timeout: true };
}

function listRuns() {
  if (!fs.existsSync(RESULTS_ROOT)) return [];
  return fs.readdirSync(RESULTS_ROOT)
    .filter(n => !n.startsWith('.') && fs.statSync(path.join(RESULTS_ROOT, n)).isDirectory())
    .sort()
    .reverse();
}

function listReportsMeta() {
  return listRuns().map(runId => {
    const runRoot = path.join(RESULTS_ROOT, runId);
    const scDirs = fs.readdirSync(runRoot).filter(n => fs.statSync(path.join(runRoot, n)).isDirectory());
    const hasVideo = scDirs.some(d => fs.existsSync(path.join(runRoot, d, 'session_recording.mp4')));
    return {
      runId,
      scenarios: scDirs,
      hasMd: fs.existsSync(path.join(runRoot, 'scorecard_report.md')),
      hasCsv: fs.existsSync(path.join(runRoot, 'scorecard.csv')),
      hasVideo
    };
  });
}

// 加载指定批次的全量评测数据（兼容旧版历史记录：缺失人工评分汇总字段时自动按现行结构重建）
function loadRunData(runId) {
  const runRoot = path.join(RESULTS_ROOT, runId);
  if (!fs.existsSync(runRoot)) return null;
  const scenarios = [];
  for (const name of fs.readdirSync(runRoot)) {
    const scDir = path.join(runRoot, name);
    if (!fs.statSync(scDir).isDirectory()) continue;
    const rawPath = path.join(scDir, 'report_raw.json');
    const evalPath = path.join(scDir, 'evaluation.json');
    const raw = fs.existsSync(rawPath) ? JSON.parse(fs.readFileSync(rawPath, 'utf8')) : null;
    let summary = fs.existsSync(evalPath) ? JSON.parse(fs.readFileSync(evalPath, 'utf8')) : null;
    if (raw && (!summary || typeof summary.reviewed_count !== 'number')) {
      try { buildRunReport(runRoot); summary = fs.existsSync(evalPath) ? JSON.parse(fs.readFileSync(evalPath, 'utf8')) : null; } catch {}
    }
    if (raw) {
      if (raw.greeting?.audio_file) raw.greeting.audio_src = toFileSrc(raw.greeting.audio_file);
      raw.full_ai_audio_src = toFileSrc(raw.full_ai_audio);
      raw.session_video_src = toFileSrc(raw.session_video);
      for (const t of raw.turns || []) {
        t.user_audio_src = toFileSrc(t.user_audio_file);
        t.ai_audio_src = toFileSrc(t.ai_audio_file);
      }
    }
    scenarios.push({ id: name, raw, summary });
  }
  return { runId, scenarios };
}

async function runScenarioBatch(scenarioId, { maxTurnsCap = 12, telephonyFilter = false } = {}) {
  if (isRunning) return;
  isRunning = true;
  abortRequested = false;
  humanTakeoverNextTurn = false;
  runAbortController = new AbortController();
  const safeMax = Math.max(2, Math.min(20, Number(maxTurnsCap) || 12));
  const runId = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const runRoot = path.join(RESULTS_ROOT, runId);
  fs.mkdirSync(runRoot, { recursive: true });

  const targets = scenarioId === 'all' ? cfg.scenarios : cfg.scenarios.filter(s => s.id === scenarioId);
  try {
    for (const sc of targets) {
      if (abortRequested) break;
      const scOutDir = path.join(runRoot, sc.id);
      fs.mkdirSync(scOutDir, { recursive: true });
      log(`开始执行评测场景：${sc.agent_name}（轮次上限 ${safeMax}${telephonyFilter ? ' · 8kHz窄带' : ''} · 人工评分+完整录屏）`);

      const startBtnSel = 'button:has-text("开始对话"), [role="button"]:has-text("开始对话"), :text-is("开始对话")';
      const endBtnSel = 'button:has-text("结束对话"), [role="button"]:has-text("结束对话"), :text-is("结束对话")';
      const runTestSel = 'button:has-text("运行测试"), [role="button"]:has-text("运行测试"), :text-is("运行测试")';

      const pageTextNow = await targetPage.evaluate(() => document.body.innerText || '');
      const isAnyTestPage = pageTextNow.includes('开始对话') || pageTextNow.includes('结束对话') || pageTextNow.includes('实时对话');
      const alreadyOnTestPage = pageTextNow.includes(sc.agent_name) && isAnyTestPage;

      if (!alreadyOnTestPage) {
        if (isAnyTestPage && !pageTextNow.includes(sc.agent_name)) {
          log(`当前位于其他智能体测试页，正在自动返回列表切换至「${sc.agent_name}」...`);
          await targetPage.goto(targetUrl, { waitUntil: 'domcontentloaded' }).catch(() => {});
          await targetPage.waitForTimeout(1500);
        }
        const agentEl = targetPage.getByText(sc.agent_name, { exact: true }).first();
        await agentEl.waitFor({ state: 'visible', timeout: 6000 }).catch(() => {});
        if (await agentEl.count()) {
          await agentEl.click({ timeout: 4000 }).catch(() => {});
          await targetPage.waitForTimeout(1200);
        }
        const runTestBtn = targetPage.locator(runTestSel).first();
        await runTestBtn.waitFor({ state: 'visible', timeout: 5000 }).catch(() => {});
        if (await runTestBtn.count()) {
          await runTestBtn.click({ timeout: 4000 }).catch(() => {});
          await targetPage.waitForTimeout(1500);
        }
      }

      const activeEndBtn = targetPage.locator(endBtnSel).first();
      if (await activeEndBtn.isVisible().catch(() => false)) {
        await activeEndBtn.click({ timeout: 4000 }).catch(() => {});
        await targetPage.waitForTimeout(1200);
      }

      const startBtn = targetPage.locator(startBtnSel).first();
      await startBtn.waitFor({ state: 'visible', timeout: 8000 }).catch(() => {});

      // 启动轻量 CDP 暂存录屏（仅在用户点击顶栏「保存完整录屏」时封装 MP4，避免 results/ 体积膨胀）
      const screenRec = createScreenRecorder();
      let screenStopped = false;
      await screenRec.start();
      try {

      let prevText = await targetPage.evaluate(() => document.body.innerText || '');
      await targetPage.evaluate(() => {
        if (window.__vaTest) window.__vaTest.peakAudioLevel = 0;
        window.__startTurnRecording && window.__startTurnRecording();
      });

      // 从磁盘同步通话过程中操作员已提前勾选的评分、标签与本场总评，避免被后续写入覆盖
      const syncDiskReviews = (rep) => {
        const p = path.join(scOutDir, 'report_raw.json');
        if (!fs.existsSync(p)) return {};
        try {
          const disk = JSON.parse(fs.readFileSync(p, 'utf8'));
          if (typeof disk.run_note === 'string') rep.run_note = disk.run_note;
          const map = {};
          if (disk.greeting?.eval?.human_review) map[0] = disk.greeting.eval.human_review;
          for (const t of disk.turns || []) {
            if (t.eval?.human_review) map[t.turn] = t.eval.human_review;
          }
          return map;
        } catch { return {}; }
      };

      const report = {
        scenario_id: sc.id,
        agent_name: sc.agent_name,
        max_turns_cap: safeMax,
        telephony_filter: telephonyFilter,
        started_at: new Date().toISOString(),
        run_note: '',
        greeting: { ttfbMs: null, text_delta: '', audio_file: null, eval: buildObjectiveTurnEval({ turn: 0 }) },
        turns: [],
        full_ai_audio: null,
        session_video: null
      };
      fs.writeFileSync(path.join(scOutDir, 'report_raw.json'), JSON.stringify(report, null, 2));
      // 提前渲染第 0 轮（开场白）评分窗口，无需等待播报结束即可直接标注
      emitUi('turn_started', {
        runId, scenarioId: sc.id, turn: 0, speakerMode: 'ai', name: '开场白', eval: report.greeting.eval
      });

      const callStartTs = Date.now();
      if (await startBtn.isVisible().catch(() => false)) {
        await startBtn.click({ timeout: 5000 }).catch(async () => {
          await startBtn.evaluate(el => el.click()).catch(() => {});
        });
        log('已点击「开始对话」，正在等待开场白...');
      } else {
        log('未自动识别到「开始对话」按钮（请确认左侧页面已进入对应智能体的「运行测试」界面）', 'warn');
      }

      const gMetrics = await waitForAiTurn(callStartTs, prevText, { maxWaitStartMs: 16000, silenceAfterMs: 1000 });
      const gB64 = await targetPage.evaluate(() => window.__stopTurnRecordingBase64 ? window.__stopTurnRecordingBase64() : '');
      const gDelta = await getLatestDelta(prevText);
      prevText = gDelta.fullText;
      const gSplit = splitTurnDelta(gDelta.delta);
      const cleanGreeting = gSplit.agentText || gDelta.delta;
      const gFile = saveWebmAndWav(gB64, scOutDir, 'turn_0_greeting');
      const gRevMap = syncDiskReviews(report);
      const gEval = buildObjectiveTurnEval({ turn: 0, ttfbMs: gMetrics.ttfbMs, aiAudioPath: gFile, existingReview: gRevMap[0] || null });

      const history = [{ role: 'ai', text: cleanGreeting }];
      let patience = 100;
      let lastTtfbMs = gMetrics.ttfbMs;

      report.greeting = { ...gMetrics, text_delta: cleanGreeting, audio_file: gFile, eval: gEval };
      fs.writeFileSync(path.join(scOutDir, 'report_raw.json'), JSON.stringify(report, null, 2));
      emitUi('turn_update', {
        runId, scenarioId: sc.id, turn: 0, speakerMode: 'ai', name: '开场白', patience,
        aiText: cleanGreeting, ttfbMs: gMetrics.ttfbMs, aiAudioSrc: toFileSrc(gFile), eval: gEval
      });

      const randCtx = rollRandomScenarioContext(sc, getSettings().randomizeScenario);
      if (randCtx.summaryText) log(`[${sc.id}] 本场随机画像：${randCtx.summaryText}`);
      const coveredIds = new Set();

      for (let turnNum = 1; turnNum <= safeMax; turnNum++) {
        if (abortRequested) break;

        // 本轮开始瞬间即提前放出评分窗口，消除等待回复的标注延迟
        const isHumanThisTurn = Boolean(humanTakeoverNextTurn);
        emitUi('turn_started', {
          runId, scenarioId: sc.id, turn: turnNum,
          speakerMode: isHumanThisTurn ? 'human' : 'ai',
          name: isHumanThisTurn ? '人工接管回合' : '准备发话...',
          progress: `${coveredIds.size}/${sc.turns.length}`,
          eval: buildObjectiveTurnEval({ turn: turnNum })
        });

        // 1-Turn Human Takeover branch: pause AI caller for THIS turn ONLY, enable physical mic, then resume AI next turn
        if (humanTakeoverNextTurn) {
          humanTakeoverNextTurn = false;
          emitUi('human_turn_state', { active: true, turn: turnNum });
          const micOk = await targetPage.evaluate(() => window.__setHumanMicActive && window.__setHumanMicActive(true));
          log(`[${sc.id} · 第 ${turnNum} 轮] 已暂停 AI 回话 1 轮${micOk ? '并开启本机麦克风，请直接开口说话' : '（麦克风授权失败）'}，说完请点击「结束接管 → 恢复AI」`, 'warn');

          const humanStartTs = Date.now();
          await targetPage.evaluate(() => window.__startTurnRecording && window.__startTurnRecording());

          await new Promise(resolve => {
            humanDoneResolver = resolve;
            const iv = setInterval(() => {
              if (abortRequested) { clearInterval(iv); resolve(); }
            }, 150);
            const orig = resolve;
            humanDoneResolver = () => { clearInterval(iv); orig(); };
          });
          humanDoneResolver = null;

          await targetPage.evaluate(() => window.__setHumanMicActive && window.__setHumanMicActive(false));
          emitUi('human_turn_state', { active: false, turn: turnNum });
          if (abortRequested) break;

          const humanEndTs = Date.now();
          log(`[${sc.id} · 第 ${turnNum} 轮] 人工接管回合结束，正在监听被测助手回复并将在下一轮自动恢复 AI 模拟。`);
          // 容错：若被测助手在操作员点击「结束接管」前已开始播报，向前放宽 4 秒监听窗口避免误超时
          const listenSinceTs = Math.max(humanStartTs, humanEndTs - 4000);
          const aiMetrics = await waitForAiTurn(listenSinceTs, prevText, { maxWaitStartMs: 16000, silenceAfterMs: 950 });
          const turnB64 = await targetPage.evaluate(() => window.__stopTurnRecordingBase64 ? window.__stopTurnRecordingBase64() : '');
          const turnFile = saveWebmAndWav(turnB64, scOutDir, `turn_${turnNum}_ai`);

          const delta = await getLatestDelta(prevText);
          prevText = delta.fullText;
          const sp = splitTurnDelta(delta.delta);
          const netAsr = await getNetAsrSince(humanStartTs);
          const humanAsr = sp.userAsrText || netAsr || '（人工语音输入）';
          const cleanAiText = sp.agentText || delta.delta;

          history.push({ role: 'user', text: humanAsr });
          history.push({ role: 'ai', text: cleanAiText });
          lastTtfbMs = aiMetrics.ttfbMs;

          const revMap = syncDiskReviews(report);
          const turnEval = buildObjectiveTurnEval({ turn: turnNum, ttfbMs: aiMetrics.ttfbMs, aiAudioPath: turnFile, existingReview: revMap[turnNum] || null });
          const turnRecord = {
            turn: turnNum, speaker_mode: 'human', name: '人工接管回合', triggers: [], reason: '人工接管1轮',
            patience, user_text: humanAsr, user_asr_text: humanAsr, user_audio_file: null,
            ai_metrics: aiMetrics, ai_text_delta: cleanAiText, ai_audio_file: turnFile, eval: turnEval
          };
          const existIdx = report.turns.findIndex(x => x.turn === turnNum);
          if (existIdx >= 0) report.turns[existIdx] = turnRecord; else report.turns.push(turnRecord);
          fs.writeFileSync(path.join(scOutDir, 'report_raw.json'), JSON.stringify(report, null, 2));

          emitUi('turn_update', {
            runId, scenarioId: sc.id, turn: turnNum, speakerMode: 'human', name: '人工接管回合',
            progress: `${coveredIds.size}/${sc.turns.length}`, patience,
            userText: humanAsr, userAsrText: humanAsr, userAudioSrc: '',
            aiText: cleanAiText, ttfbMs: aiMetrics.ttfbMs, aiAudioSrc: toFileSrc(turnFile), eval: turnEval
          });
          continue;
        }

        const latestAiText = history[history.length - 1]?.text || '';
        const tLlm0 = Date.now();
        const plan = await decideNextCallerTurn({
          basePrompt: cfg.base_caller_system_prompt, sc, latestAiText, history,
          objectives: sc.turns, coveredIds, turnNum, maxTurns: safeMax,
          patience, lastTtfbMs, latestDefects: [], randomPromptAddon: randCtx.promptAddon,
          signal: runAbortController?.signal
        });
        if (abortRequested) { emitUi('turn_cancelled', { scenarioId: sc.id, turn: turnNum }); break; }
        const llmMs = Date.now() - tLlm0;
        patience = Math.max(0, Math.min(100, patience + (plan.patience_delta || 0)));
        for (const cid of (plan.covered_now || [])) coveredIds.add(cid);

        if (plan.should_end && !plan.utterance_ja) {
          emitUi('turn_cancelled', { scenarioId: sc.id, turn: turnNum });
          log(`[${sc.id}] 评测目标已收口或对方已结束通话（共 ${turnNum - 1} 轮）`);
          break;
        }

        // 模拟用户文本生成后立即回填至已展开的评分卡片中
        emitUi('turn_user_ready', {
          runId, scenarioId: sc.id, turn: turnNum, name: plan.target_name,
          progress: `${coveredIds.size}/${sc.turns.length}`, userText: plan.utterance_ja
        });

        if (plan.mode === 'silence_first') {
          log(`[${sc.id} · 第 ${turnNum} 轮] 静默 6.5s 检测超时提醒...`);
          await targetPage.evaluate(() => window.__startTurnRecording && window.__startTurnRecording());
          const silUntil = Date.now() + 6500;
          while (Date.now() < silUntil && !abortRequested) await targetPage.waitForTimeout(200);
          if (abortRequested) break;
          const silB64 = await targetPage.evaluate(() => window.__stopTurnRecordingBase64 ? window.__stopTurnRecordingBase64() : '');
          saveWebmAndWav(silB64, scOutDir, `turn_${turnNum}_silence_check`);
        }

        if (plan.mode !== 'barge_in') {
          const waitSpeakEnd = Date.now() + 8000;
          while (Date.now() < waitSpeakEnd && !abortRequested) {
            const speaking = await targetPage.evaluate(() => Boolean(window.__vaTest?.aiSpeaking)).catch(() => false);
            if (!speaking) break;
            await targetPage.waitForTimeout(120);
          }
        }

        const turnPlayStartTs = Date.now();
        const dynWav = path.join(scOutDir, `turn_${turnNum}_user.wav`);
        await targetPage.evaluate((tel) => window.__beginUserAudioStream && window.__beginUserAudioStream(tel), telephonyFilter);
        const ttsRes = await streamSynthesizeSpeech(plan.utterance_ja, dynWav, {
          telephonyFilter,
          signal: runAbortController?.signal,
          onChunk: async (chunkB64, sr) => {
            if (abortRequested) return;
            await targetPage.evaluate(([d, r]) => window.__pushUserPcmChunk && window.__pushUserPcmChunk(d, r), [chunkB64, sr]);
          }
        });
        if (abortRequested) break;
        let playInfo;
        if (ttsRes.streamed) {
          playInfo = await targetPage.evaluate(() => window.__finishUserAudioStream());
        } else {
          const b64 = fs.readFileSync(dynWav).toString('base64');
          playInfo = await targetPage.evaluate((data) => window.__playUserWavBase64(data), b64);
        }
        if (abortRequested) break;
        log(`[${sc.id} · 第 ${turnNum} 轮] ${plan.target_name}（LLM ${llmMs}ms/${plan.model || ''} · TTS首包 ${ttsRes.ttfaMs}ms · ${ttsRes.engine}）`);
        history.push({ role: 'user', text: plan.utterance_ja });

        await targetPage.evaluate(() => window.__startTurnRecording && window.__startTurnRecording());
        const aiMetrics = await waitForAiTurn(playInfo.endTs, prevText, {
          maxWaitStartMs: 16000, silenceAfterMs: 950,
          stopAfterStartMs: plan.mode === 'barge_in' ? 1800 : 0
        });
        const turnB64 = await targetPage.evaluate(() => window.__stopTurnRecordingBase64 ? window.__stopTurnRecordingBase64() : '');
        const turnFile = saveWebmAndWav(turnB64, scOutDir, `turn_${turnNum}_ai`);

        const delta = await getLatestDelta(prevText);
        prevText = delta.fullText;
        const sp = splitTurnDelta(delta.delta);
        const netAsr = await getNetAsrSince(turnPlayStartTs);
        const userAsrText = sp.userAsrText || netAsr || '';
        const cleanAiText = sp.agentText || delta.delta;

        history.push({ role: 'ai', text: cleanAiText });
        lastTtfbMs = aiMetrics.ttfbMs;
        if (lastTtfbMs > 2500) patience = Math.max(0, patience - 15);

        const revMap = syncDiskReviews(report);
        const turnEval = buildObjectiveTurnEval({ turn: turnNum, ttfbMs: aiMetrics.ttfbMs, aiAudioPath: turnFile, existingReview: revMap[turnNum] || null });
        const turnRecord = {
          turn: turnNum, speaker_mode: 'ai', name: plan.target_name, triggers: plan.triggers, reason: plan.reason || '',
          patience, user_text: plan.utterance_ja, user_asr_text: userAsrText, user_audio_file: dynWav,
          ai_metrics: aiMetrics, ai_text_delta: cleanAiText, ai_audio_file: turnFile, eval: turnEval
        };
        const existIdx = report.turns.findIndex(x => x.turn === turnNum);
        if (existIdx >= 0) report.turns[existIdx] = turnRecord; else report.turns.push(turnRecord);
        fs.writeFileSync(path.join(scOutDir, 'report_raw.json'), JSON.stringify(report, null, 2));

        emitUi('turn_update', {
          runId, scenarioId: sc.id, turn: turnNum, speakerMode: 'ai', name: plan.target_name, triggers: plan.triggers,
          progress: `${coveredIds.size}/${sc.turns.length}`, patience,
          userText: plan.utterance_ja, userAsrText, userAudioSrc: toFileSrc(dynWav),
          aiText: cleanAiText, ttfbMs: aiMetrics.ttfbMs, aiAudioSrc: toFileSrc(turnFile), eval: turnEval
        });

        if (plan.should_end || !aiMetrics.responded || patience < 15) {
          log(`[${sc.id}] 本场通话结束（共 ${turnNum} 轮，耐心余量 ${patience}）`);
          break;
        }
      }

      await targetPage.waitForTimeout(500);
      const fullB64 = await targetPage.evaluate(() => window.__stopFullRecordingBase64 ? window.__stopFullRecordingBase64() : '');
      report.full_ai_audio = saveWebmAndWav(fullB64, scOutDir, 'full_ai_audio');

      const endBtn = targetPage.locator('button:has-text("结束对话"), [role="button"]:has-text("结束对话"), :text-is("结束对话")').first();
      if (await endBtn.isVisible().catch(() => false)) await endBtn.click({ timeout: 3000 }).catch(() => {});

      const cap = await screenRec.stopCapture();
      screenStopped = true;
      if (cap.frames.length >= 2) {
        lastScreenDraft = { runId, runRoot, scOutDir, scenarioId: sc.id, framesDir: cap.framesDir, frames: cap.frames, audioWavPath: report.full_ai_audio };
        emitUi('recording_ready', { runId, scenarioId: sc.id, frameCount: cap.frames.length });
      } else {
        try { fs.rmSync(cap.framesDir, { recursive: true, force: true }); } catch {}
      }
      await targetPage.screenshot({ path: path.join(scOutDir, 'final_screen.png'), fullPage: true }).catch(() => {});
      fs.writeFileSync(path.join(scOutDir, 'report_raw.json'), JSON.stringify(report, null, 2));
      if (!abortRequested) await targetPage.waitForTimeout(600);
      } finally {
        if (!screenStopped) {
          const cap = await screenRec.stopCapture().catch(() => null);
          if (cap?.framesDir) try { fs.rmSync(cap.framesDir, { recursive: true, force: true }); } catch {}
        }
      }
    }

    buildRunReport(runRoot);
    if (abortRequested) {
      emitUi('run_stopped', { runId, data: loadRunData(runId), reports: listReportsMeta() });
      log('已停止当前评测，音频与结构化报告已保存；如需视频证据可点击顶栏「保存完整录屏」。', 'warn');
    } else {
      emitUi('run_completed', { runId, data: loadRunData(runId), reports: listReportsMeta() });
      log(`通话已完成，音频与报告已保存至 results/${runId}/；如需留存视频可点击顶栏「保存完整录屏」。`);
    }
  } catch (e) {
    if (abortRequested) {
      emitUi('run_stopped', { runId });
      log('已立即停止当前评测。', 'warn');
    } else {
      log(`执行出错: ${e.message}`, 'error');
      emitUi('run_error', { error: e.message });
    }
  } finally {
    isRunning = false;
    humanTakeoverNextTurn = false;
    runAbortController = null;
    if (targetPage && !targetPage.isClosed()) {
      await targetPage.evaluate(() => window.__setHumanMicActive && window.__setHumanMicActive(false)).catch(() => {});
    }
  }
}

(async () => {
  const profileDir = path.join(ROOT_DIR, 'chrome_profile');
  const launchOpts = {
    headless: false,
    viewport: null,
    permissions: ['microphone'],
    args: [
      '--window-size=1220,760',
      '--autoplay-policy=no-user-gesture-required',
      '--disable-background-timer-throttling',
      '--disable-renderer-backgrounding',
      '--disable-backgrounding-occluded-windows'
    ]
  };
  let context;
  try { context = await chromium.launchPersistentContext(profileDir, { ...launchOpts, channel: 'chrome' }); }
  catch { context = await chromium.launchPersistentContext(profileDir, launchOpts); }

  await context.addInitScript(INIT_SCRIPT);
  targetPage = context.pages()[0] || await context.newPage();
  await targetPage.goto(targetUrl, { waitUntil: 'domcontentloaded' }).catch(() => {});
  try { cdpSession = await context.newCDPSession(targetPage); } catch {}





  uiPage = await context.newPage();
  await uiPage.exposeFunction('appGetInitData', () => {
    const st = getSettings();
    return {
      cfg, settings: { ...st, targetUrl }, issueTagPresets: ISSUE_TAG_PRESETS, issueTagSpecs: ISSUE_TAG_SPECS,
      llmModel: st.callerModel, ttsModel: st.ttsModel, targetUrl, reports: listReportsMeta()
    };
  });
  await uiPage.exposeFunction('appGetSettings', () => ({ ...getSettings(), targetUrl }));
  await uiPage.exposeFunction('appSaveSettings', async (patch) => {
    const updated = updateSettings(patch);
    const prevUrl = targetUrl;
    if (typeof patch?.targetUrl === 'string' && patch.targetUrl.trim()) targetUrl = patch.targetUrl.trim();
    persistEnvSettings(updated, targetUrl);
    if (targetUrl !== prevUrl && targetPage && !targetPage.isClosed()) {
      await targetPage.goto(targetUrl, { waitUntil: 'domcontentloaded' }).catch(() => {});
    }
    log(`系统设置已保存并立即生效（提供商=${updated.provider} · 对话=${updated.callerModel} · 语音=${updated.ttsModel}）`);
    return { ...updated, targetUrl };
  });
  await uiPage.exposeFunction('appStartScenario', (sid, opts) => { runScenarioBatch(sid, opts); return true; });
  await uiPage.exposeFunction('appRequestHumanTakeover', () => {
    if (!isRunning) return false;
    humanTakeoverNextTurn = true;
    log('已预约：下一回合暂停 AI 回话，切换为人工插话（仅暂停 1 轮，之后自动恢复）', 'warn');
    return true;
  });
  await uiPage.exposeFunction('appFinishHumanTurn', () => {
    if (humanDoneResolver) humanDoneResolver();
    return true;
  });
  await uiPage.exposeFunction('appStopScenario', async () => {
    abortRequested = true;
    if (humanDoneResolver) humanDoneResolver();
    try { runAbortController?.abort(); } catch {}
    if (targetPage && !targetPage.isClosed()) {
      await targetPage.evaluate(() => {
        window.__stopUserAudioNow && window.__stopUserAudioNow();
        window.__setHumanMicActive && window.__setHumanMicActive(false);
      }).catch(() => {});
      const endBtn = targetPage.locator('button:has-text("结束对话"), [role="button"]:has-text("结束对话"), :text-is("结束对话")').first();
      if (await endBtn.isVisible().catch(() => false)) await endBtn.click({ timeout: 2000 }).catch(() => {});
    }
    isRunning = false;
    emitUi('run_stopped', {});
    log('已中止测试请求并挂断通话', 'warn');
    return true;
  });
  await uiPage.exposeFunction('appListReports', () => listReportsMeta());
  await uiPage.exposeFunction('appLoadRun', (runId) => loadRunData(runId));
  await uiPage.exposeFunction('appGetMarkdownReport', (runId) => {
    const mdPath = path.join(RESULTS_ROOT, runId, 'scorecard_report.md');
    if (!fs.existsSync(mdPath)) return { exists: false, content: '' };
    return { exists: true, content: fs.readFileSync(mdPath, 'utf8') };
  });
  await uiPage.exposeFunction('appSaveHumanReview', (runId, scenarioId, turnNum, review) => {
    saveHumanReview(path.join(RESULTS_ROOT, runId), scenarioId, turnNum, review);
    const techList = inferSuspectedTechIssues(review.tags || [], []);
    const techHint = techList.length ? ` ｜ 疑似技术问题：${techList.join('；')}` : '';
    log(`已保存人工评分：${scenarioId} 第 ${turnNum} 轮（${review.score != null ? review.score + '分' : '更新标签/备注'}）${techHint}`);
    return loadRunData(runId);
  });
  await uiPage.exposeFunction('appSaveRunNote', (runId, scenarioId, note) => {
    saveRunNote(path.join(RESULTS_ROOT, runId), scenarioId, note);
    log(`已保存本场人工总评（${scenarioId}），Markdown/CSV 报告已同步刷新`);
    return loadRunData(runId);
  });
  await uiPage.exposeFunction('appSavePrompts', (basePrompt, sid, scenePrompt) => {
    cfg.base_caller_system_prompt = basePrompt;
    const sc = cfg.scenarios.find(s => s.id === sid);
    if (sc) sc.scenario_persona_prompt = scenePrompt;
    fs.writeFileSync(CFG_PATH, JSON.stringify(cfg, null, 2));
    return cfg;
  });
  await uiPage.exposeFunction('appClickTarget', async (rx, ry) => {
    if (!targetPage || targetPage.isClosed()) return;
    const vp = await targetPage.evaluate(() => ({ width: window.innerWidth, height: window.innerHeight })).catch(() => ({ width: 1220, height: 760 }));
    await targetPage.mouse.click(Math.round(rx * vp.width), Math.round(ry * vp.height)).catch(() => {});
  });
  await uiPage.exposeFunction('appReloadTarget', async () => {
    if (targetPage && !targetPage.isClosed()) await targetPage.reload({ waitUntil: 'domcontentloaded' }).catch(() => {});
  });
  await uiPage.exposeFunction('appFocusTarget', async () => {
    if (targetPage && !targetPage.isClosed()) {
      await targetPage.bringToFront().catch(() => {});
    }
  });
  await uiPage.exposeFunction('appSaveLastRecording', async () => {
    if (!lastScreenDraft) return { ok: false, msg: '当前没有可导出的临时录屏（请先完成一次评测）' };
    log(`正在将本场录屏封装为 MP4（共 ${lastScreenDraft.frames.length} 帧）...`);
    const { runId, runRoot, scOutDir } = lastScreenDraft;
    const mp4Path = muxDraftToMp4(lastScreenDraft);
    clearLastScreenDraft();
    if (!mp4Path) {
      log('录屏封装失败，请检查 ffmpeg 是否可用', 'error');
      return { ok: false, msg: '录屏封装失败' };
    }
    const rawPath = path.join(scOutDir, 'report_raw.json');
    if (fs.existsSync(rawPath)) {
      const raw = JSON.parse(fs.readFileSync(rawPath, 'utf8'));
      raw.session_video = mp4Path;
      fs.writeFileSync(rawPath, JSON.stringify(raw, null, 2));
    }
    buildRunReport(runRoot);
    log(`完整录屏已保存至 results/${runId}/${path.basename(scOutDir)}/session_recording.mp4`);
    return { ok: true, runId, data: loadRunData(runId) };
  });
  await uiPage.exposeFunction('appOpenResultsFolder', (runId) => {
    const dir = runId ? path.join(RESULTS_ROOT, runId) : RESULTS_ROOT;
    fs.mkdirSync(dir, { recursive: true });
    try {
      if (process.platform === 'darwin') execFileSync('open', [dir]);
      else if (process.platform === 'win32') execFileSync('explorer', [dir]);
      else execFileSync('xdg-open', [dir]);
    } catch {}
  });

  await uiPage.goto(pathToFileURL(path.join(__dirname, 'ui', 'index.html')).href, { waitUntil: 'domcontentloaded' });
  await uiPage.bringToFront();

  // 必须在 uiPage 初始化完成后再注册弹窗拦截器，避免把工作台自身页面误关
  context.on('page', async (newPage) => {
    if (newPage === uiPage || newPage === targetPage) return;
    try {
      await newPage.waitForLoadState('domcontentloaded', { timeout: 3000 }).catch(() => {});
      const popUrl = newPage.url();
      if (popUrl.includes('/src/ui/index.html')) return;
      await newPage.close().catch(() => {});
      if (popUrl && popUrl !== 'about:blank' && targetPage && !targetPage.isClosed()) {
        await targetPage.goto(popUrl, { waitUntil: 'domcontentloaded' }).catch(() => {});
        log(`已将新弹窗页面锁定在内嵌视图内打开：${popUrl}`);
      }
      if (uiPage && !uiPage.isClosed()) await uiPage.bringToFront().catch(() => {});
    } catch {}
  });

  // Flicker-free sequential CDP mirror loop with frame deduplication
  let mirrorStopped = false;
  let lastFrameB64 = '';

  (async function mirrorLoop() {
    while (!mirrorStopped) {
      if (targetPage && !targetPage.isClosed() && uiPage && !uiPage.isClosed()) {
        try {
          const vp = await targetPage.evaluate(() => ({ width: window.innerWidth, height: window.innerHeight })).catch(() => ({ width: 1220, height: 760 }));
          let b64 = '';
          if (cdpSession) {
            const res = await cdpSession.send('Page.captureScreenshot', { format: 'jpeg', quality: 70 });
            b64 = res?.data || '';
          } else {
            const buf = await targetPage.screenshot({ type: 'jpeg', quality: 70 });
            b64 = buf.toString('base64');
          }
          if (b64 && b64 !== lastFrameB64) {
            lastFrameB64 = b64;
            emitUi('frame', { img: `data:image/jpeg;base64,${b64}`, url: targetPage.url(), width: vp.width, height: vp.height });
          }
        } catch {}
      }
      await new Promise(r => setTimeout(r, 600));
    }
  })();

  log('Voice Agent QA Studio 已就绪（人工快速评分 + 1轮人工接管 + ASR实识对照 + 按需保存录屏）。');
  uiPage.on('close', () => { mirrorStopped = true; clearLastScreenDraft(); context.close().catch(() => {}); });
})();
