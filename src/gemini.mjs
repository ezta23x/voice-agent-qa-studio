import fs from 'fs';
import { execFileSync } from 'child_process';

const FFMPEG_BIN = ['/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg'].find(p => fs.existsSync(p)) || 'ffmpeg';

// Mutable runtime configuration (hot-updated via Settings UI without restart)
const runtimeConfig = {
  provider: process.env.API_PROVIDER || (process.env.VERTEX_API_KEY ? 'vertex' : (process.env.GEMINI_API_KEY ? 'gemini' : 'vertex')),
  vertexApiKey: process.env.VERTEX_API_KEY || '',
  geminiApiKey: process.env.GEMINI_API_KEY || '',
  callerModel: process.env.VERTEX_MODEL || 'gemini-3.5-flash-lite',
  ttsModel: process.env.VERTEX_TTS_MODEL || 'gemini-3.8-flash-tts',
  ttsVoice: process.env.VERTEX_TTS_VOICE || 'Kore',
  evalModel: process.env.EVAL_MODEL || 'gemini-3.8-flash',
  evalThinkingLevel: process.env.EVAL_THINKING_LEVEL || 'HIGH',
  randomizeScenario: process.env.RANDOMIZE_SCENARIO !== 'false'
};

// Backwards-compatible exports (initial defaults)
export const VERTEX_MODEL = runtimeConfig.callerModel;
export const EVAL_MODEL = runtimeConfig.evalModel;
export const VERTEX_TTS_MODEL = runtimeConfig.ttsModel;

let cachedLlm = null; // { model, fastTc }
let cachedTts = null; // { model, supportsStream }

export function getSettings() {
  return { ...runtimeConfig };
}

export function updateSettings(patch = {}) {
  if (patch.provider === 'vertex' || patch.provider === 'gemini') runtimeConfig.provider = patch.provider;
  if (typeof patch.vertexApiKey === 'string') runtimeConfig.vertexApiKey = patch.vertexApiKey.trim();
  if (typeof patch.geminiApiKey === 'string') runtimeConfig.geminiApiKey = patch.geminiApiKey.trim();
  if (typeof patch.callerModel === 'string' && patch.callerModel.trim()) runtimeConfig.callerModel = patch.callerModel.trim();
  if (typeof patch.ttsModel === 'string' && patch.ttsModel.trim()) runtimeConfig.ttsModel = patch.ttsModel.trim();
  if (typeof patch.ttsVoice === 'string' && patch.ttsVoice.trim()) runtimeConfig.ttsVoice = patch.ttsVoice.trim();
  if (typeof patch.evalModel === 'string' && patch.evalModel.trim()) runtimeConfig.evalModel = patch.evalModel.trim();
  if (typeof patch.evalThinkingLevel === 'string' && patch.evalThinkingLevel.trim()) runtimeConfig.evalThinkingLevel = patch.evalThinkingLevel.trim().toUpperCase();
  if (typeof patch.randomizeScenario === 'boolean') runtimeConfig.randomizeScenario = patch.randomizeScenario;
  // Reset cached endpoint capabilities so new provider/models take effect immediately
  cachedLlm = null;
  cachedTts = null;
  return getSettings();
}

function getActiveApiKey() {
  if (runtimeConfig.provider === 'gemini') return runtimeConfig.geminiApiKey || runtimeConfig.vertexApiKey;
  return runtimeConfig.vertexApiKey || runtimeConfig.geminiApiKey;
}

function getLlmFallbacks() {
  return [
    runtimeConfig.callerModel,
    'gemini-3.5-flash-lite',
    'gemini-3.8-flash',
    'gemini-3.6-flash',
    'gemini-3-flash-preview'
  ].filter((v, i, a) => v && a.indexOf(v) === i);
}

function getTtsFallbacks() {
  return [
    runtimeConfig.ttsModel,
    'gemini-3.8-flash-tts',
    'gemini-3.8-flash-lite-tts',
    'gemini-3.1-flash-tts-preview',
    'gemini-2.5-flash-preview-tts'
  ].filter((v, i, a) => v && a.indexOf(v) === i);
}

export function buildGeminiEndpoint(model, stream = false) {
  const method = stream ? 'streamGenerateContent?alt=sse&' : 'generateContent?';
  const useGeminiApi = runtimeConfig.provider === 'gemini' || (!runtimeConfig.vertexApiKey && Boolean(runtimeConfig.geminiApiKey));
  if (!useGeminiApi && runtimeConfig.vertexApiKey) {
    return `https://aiplatform.googleapis.com/v1/publishers/google/models/${encodeURIComponent(model)}:${method}key=${encodeURIComponent(runtimeConfig.vertexApiKey)}`;
  }
  const key = runtimeConfig.geminiApiKey || runtimeConfig.vertexApiKey;
  if (key) {
    return `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:${method}key=${encodeURIComponent(key)}`;
  }
  throw new Error('未配置 API Key，请前往「系统设置」填写 Vertex AI 或 Gemini API Key');
}

export function sanitizeSpokenText(raw) {
  return String(raw || '')
    .replace(/^(?:顧客|ユーザー|お客様|発話|セリフ|User|Customer)\s*[:：]\s*/gim, '')
    .replace(/[（(][^）)]{0,45}[）)]/g, '')
    .replace(/[【\[][^】\]]{0,45}[】\]]/g, '')
    .replace(/[*_`#"'「」『』]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

export function rollRandomScenarioContext(sc, enabled = true) {
  if (!enabled || !sc?.random_pool || typeof sc.random_pool !== 'object') {
    return { summaryText: '', promptAddon: '' };
  }
  const picked = [];
  for (const [k, arr] of Object.entries(sc.random_pool)) {
    if (Array.isArray(arr) && arr.length > 0) {
      const choice = arr[Math.floor(Math.random() * arr.length)];
      picked.push(choice);
    }
  }
  if (!picked.length) return { summaryText: '', promptAddon: '' };
  return {
    summaryText: picked.join(' ｜ '),
    promptAddon: `【今回の通話ランダム設定（必ずこの設定で演じること）】\n${picked.map(x => `- ${x}`).join('\n')}`
  };
}

function pcm16ToWavBuffer(pcmBuf, sampleRate = 24000, numChannels = 1) {
  const header = Buffer.alloc(44);
  header.write('RIFF', 0);
  header.writeUInt32LE(36 + pcmBuf.length, 4);
  header.write('WAVE', 8);
  header.write('fmt ', 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(numChannels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(sampleRate * numChannels * 2, 28);
  header.writeUInt16LE(numChannels * 2, 32);
  header.writeUInt16LE(16, 34);
  header.write('data', 36);
  header.writeUInt32LE(pcmBuf.length, 40);
  return Buffer.concat([header, pcmBuf]);
}

function makeSignal(userSignal, timeoutMs = 10000) {
  const ts = AbortSignal.timeout(timeoutMs);
  return userSignal ? AbortSignal.any([userSignal, ts]) : ts;
}

export async function callGeminiJson({
  systemPrompt, contents, temperature = 0.2, maxOutputTokens = 2048,
  disableThinking = false, thinkingLevel = null, modelOverride = null, timeoutMs = null, signal
} = {}) {
  let lastErr = null;
  const fallbacks = getLlmFallbacks();
  const baseList = modelOverride
    ? [modelOverride, ...fallbacks.filter(m => m !== modelOverride)]
    : (cachedLlm ? [cachedLlm.model, ...fallbacks.filter(m => m !== cachedLlm.model)] : fallbacks);
  const effectiveTimeout = timeoutMs || (thinkingLevel === 'HIGH' ? 28000 : (disableThinking ? 7000 : 12000));

  for (const model of baseList) {
    if (signal?.aborted) throw new Error('用户已中止请求');
    let attempts;
    if (thinkingLevel) {
      attempts = [{ thinkingLevel }, {}];
    } else if (disableThinking) {
      // Prefer thinkingLevel:'LOW' (supported by 3.5/3.8) over thinkingBudget:0
      attempts = cachedLlm?.model === model && cachedLlm.fastTc
        ? [cachedLlm.fastTc]
        : [{ thinkingLevel: 'LOW' }, { thinkingBudget: 0 }, {}];
    } else {
      attempts = [{}];
    }
    for (const tc of attempts) {
      if (signal?.aborted) throw new Error('用户已中止请求');
      try {
        const url = buildGeminiEndpoint(model);
        const generationConfig = { temperature, maxOutputTokens, responseMimeType: 'application/json' };
        if (Object.keys(tc).length) generationConfig.thinkingConfig = tc;
        const body = {
          systemInstruction: { parts: [{ text: systemPrompt }] },
          contents,
          generationConfig
        };
        const resp = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: makeSignal(signal, effectiveTimeout)
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status} (${model}): ${(await resp.text()).slice(0, 200)}`);
        const data = await resp.json();
        const raw = (data.candidates?.[0]?.content?.parts || []).map(p => p.text || '').join('').trim();
        const parsed = JSON.parse(raw);
        if (!modelOverride && !thinkingLevel && disableThinking) {
          cachedLlm = { model, fastTc: tc };
        }
        return { model, thinkingLevel: tc.thinkingLevel || null, result: parsed };
      } catch (e) {
        if (signal?.aborted) throw new Error('用户已中止请求');
        lastErr = e;
      }
    }
  }
  throw lastErr || new Error('Gemini API 调用失败');
}

export async function synthesizeSpeech(text, outWavPath, { telephonyFilter = false, fastLocalTts = false, rate = 192, signal } = {}) {
  const cleanText = sanitizeSpokenText(text) || String(text || '').trim();
  const rawWavPath = telephonyFilter ? outWavPath.replace(/\.wav$/i, '.clean.wav') : outWavPath;
  let usedEngine = 'macos-kyoko';

  const useCloudTts = !fastLocalTts && Boolean(getActiveApiKey());
  if (useCloudTts) {
    const body = {
      contents: [{ role: 'user', parts: [{ text: cleanText }] }],
      generationConfig: {
        responseModalities: ['AUDIO'],
        speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: runtimeConfig.ttsVoice } } }
      }
    };
    let ok = false;
    const fallbacks = getTtsFallbacks();
    const ttsOrder = cachedTts ? [cachedTts.model, ...fallbacks.filter(m => m !== cachedTts.model)] : fallbacks;
    for (const ttsModel of ttsOrder) {
      if (signal?.aborted) throw new Error('用户已中止请求');
      try {
        const resp = await fetch(buildGeminiEndpoint(ttsModel), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: makeSignal(signal, 10000)
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        const part = (data.candidates?.[0]?.content?.parts || []).find(p => p.inlineData?.data);
        if (!part) throw new Error('no inlineData');
        const mime = part.inlineData.mimeType || 'audio/L16;codec=pcm;rate=24000';
        const rawBuf = Buffer.from(part.inlineData.data, 'base64');
        if (mime.includes('wav')) fs.writeFileSync(rawWavPath, rawBuf);
        else {
          const m = mime.match(/rate=(\d+)/);
          fs.writeFileSync(rawWavPath, pcm16ToWavBuffer(rawBuf, m ? parseInt(m[1], 10) : 24000, 1));
        }
        usedEngine = ttsModel;
        cachedTts = { model: ttsModel, supportsStream: cachedTts?.model === ttsModel ? cachedTts.supportsStream : false };
        ok = true;
        break;
      } catch (e) {
        if (signal?.aborted) throw new Error('用户已中止请求');
      }
    }
    if (!ok) {
      const aiffPath = rawWavPath.replace(/\.wav$/i, '.aiff');
      execFileSync('say', ['-v', 'Kyoko', '-r', String(rate), '-o', aiffPath, cleanText]);
      execFileSync(FFMPEG_BIN, ['-y', '-loglevel', 'error', '-i', aiffPath, '-ar', '48000', '-ac', '1', '-c:a', 'pcm_s16le', rawWavPath]);
      if (fs.existsSync(aiffPath)) fs.unlinkSync(aiffPath);
    }
  } else {
    const aiffPath = rawWavPath.replace(/\.wav$/i, '.aiff');
    execFileSync('say', ['-v', 'Kyoko', '-r', String(rate), '-o', aiffPath, cleanText]);
    execFileSync(FFMPEG_BIN, ['-y', '-loglevel', 'error', '-i', aiffPath, '-ar', '48000', '-ac', '1', '-c:a', 'pcm_s16le', rawWavPath]);
    if (fs.existsSync(aiffPath)) fs.unlinkSync(aiffPath);
  }

  if (telephonyFilter) {
    try {
      execFileSync(FFMPEG_BIN, [
        '-y', '-loglevel', 'error', '-i', rawWavPath,
        '-af', 'highpass=f=300,lowpass=f=3400,aresample=8000,aresample=48000',
        '-ac', '1', '-c:a', 'pcm_s16le', outWavPath
      ]);
      if (fs.existsSync(rawWavPath)) fs.unlinkSync(rawWavPath);
      usedEngine += '+8kHz';
    } catch {
      fs.renameSync(rawWavPath, outWavPath);
    }
  }
  return usedEngine;
}

// Cloud SSE streaming TTS (streamGenerateContent?alt=sse -> raw 24kHz 16-bit PCM chunks)
export async function streamSynthesizeSpeech(text, outWavPath, { telephonyFilter = false, onChunk, signal } = {}) {
  const cleanText = sanitizeSpokenText(text) || String(text || '').trim();
  const body = {
    contents: [{ role: 'user', parts: [{ text: cleanText }] }],
    generationConfig: {
      responseModalities: ['AUDIO'],
      speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: runtimeConfig.ttsVoice } } }
    }
  };

  const tryStream = !(cachedTts && cachedTts.supportsStream === false);
  if (tryStream && getActiveApiKey()) {
    const ttsOrder = cachedTts ? [cachedTts.model] : getTtsFallbacks();
    for (const ttsModel of ttsOrder) {
      if (signal?.aborted) throw new Error('用户已中止请求');
      const t0 = Date.now();
      let ttfaMs = null;
      let sampleRate = 24000;
      const pcmChunks = [];
      let chunkCount = 0;
      try {
        const resp = await fetch(buildGeminiEndpoint(ttsModel, true), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: makeSignal(signal, 8000)
        });
        if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';

        const processEvent = async (dataStr) => {
          if (!dataStr || dataStr === '[DONE]') return;
          let obj;
          try { obj = JSON.parse(dataStr); } catch { return; }
          const parts = obj.candidates?.[0]?.content?.parts || [];
          for (const p of parts) {
            const b64 = p.inlineData?.data;
            if (!b64) continue;
            const mime = p.inlineData?.mimeType || 'audio/l16;rate=24000';
            const m = mime.match(/rate=(\d+)/i);
            if (m) sampleRate = parseInt(m[1], 10);
            let raw = Buffer.from(b64, 'base64');
            if (chunkCount === 0 && raw.length > 44 && raw.subarray(0, 4).toString('ascii') === 'RIFF') {
              raw = raw.subarray(44);
            }
            if (!raw.length) continue;
            if (ttfaMs === null) ttfaMs = Date.now() - t0;
            chunkCount++;
            pcmChunks.push(raw);
            if (onChunk) await onChunk(raw.toString('base64'), sampleRate);
          }
        };

        while (true) {
          if (signal?.aborted) { try { reader.cancel(); } catch {} throw new Error('用户已中止请求'); }
          const { value, done } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split(/\r?\n/);
          buf = lines.pop() || '';
          for (const line of lines) {
            if (line.startsWith('data:')) await processEvent(line.slice(5).trim());
          }
        }
        if (buf.trim().startsWith('data:')) await processEvent(buf.trim().slice(5).trim());

        if (pcmChunks.length > 0) {
          const fullPcm = Buffer.concat(pcmChunks);
          fs.writeFileSync(outWavPath, pcm16ToWavBuffer(fullPcm, sampleRate, 1));
          cachedTts = { model: ttsModel, supportsStream: true };
          return {
            engine: `${ttsModel}(流式${telephonyFilter ? '+8kHz' : ''})`,
            streamed: true,
            ttfaMs,
            totalMs: Date.now() - t0,
            chunks: chunkCount
          };
        }
      } catch (e) {
        if (signal?.aborted) throw new Error('用户已中止请求');
      }
    }
  }

  const t0 = Date.now();
  const usedEngine = await synthesizeSpeech(cleanText, outWavPath, { telephonyFilter, fastLocalTts: false, signal });
  return { engine: usedEngine, streamed: false, ttfaMs: Date.now() - t0, totalMs: Date.now() - t0, chunks: 1 };
}

export async function decideNextCallerTurn({
  basePrompt, sc, latestAiText, history, objectives, coveredIds,
  turnNum, maxTurns, patience, lastTtfbMs, latestDefects, randomPromptAddon = '', signal
}) {
  const remaining = objectives.filter(o => !coveredIds.has(o.turn));
  const fallback = remaining[0] || objectives[objectives.length - 1];

  if (!getActiveApiKey()) {
    return {
      utterance_ja: fallback?.text || 'はい、ありがとうございます。失礼いたします。',
      target_id: fallback?.turn || turnNum,
      target_name: fallback?.name || '自然收尾',
      triggers: fallback?.triggers || [],
      mode: fallback?.mode || 'normal',
      should_end: remaining.length === 0,
      covered_now: fallback ? [fallback.turn] : [],
      patience_delta: 0
    };
  }

  const systemPrompt = [
    basePrompt || '',
    sc.scenario_persona_prompt || '',
    randomPromptAddon || '',
    `あなたは「実際の日本人顧客」のように振る舞いながら、テスト観点チェックリストを自然な電話会話で検証する自律型テスターです。`,
    `必ず以下のJSONのみを出力してください：\n{\n  "utterance_ja": "電話口でそのまま話す簡潔で自然な日本語1文（15〜45文字程度・括弧や解説禁止）",\n  "target_id": 1,\n  "target_name": "今回狙う観点名",\n  "covered_ids": [1],\n  "mode": "normal | barge_in | silence_first",\n  "patience_delta": 0,\n  "should_end": false,\n  "reason": "判断理由（中国語で15字以内）"\n}\n判断基準：\n- 相手が終話挨拶をした、または全観点を検証し終えた、または耐心値が20未満で会話継続不可なら should_end=true。\n- 今回のランダム設定（姓氏・電話番号・日時・要望・追加テスト）を自然に使い、毎回同じ台詞にしないこと。\n- 直前の応答遅延が2500ms超え・同じ質問の繰り返し・明らかな誤認があれば patience_delta を -10〜-25 にし、リアルな戸惑いや急ぎ気味な反応を台詞に反映してよい。`
  ].filter(Boolean).join('\n\n');

  const checklist = objectives
    .filter(o => !coveredIds.has(o.turn))
    .map(o => `${o.turn}.${o.name}(${o.mode || 'normal'}): ${o.text}`)
    .join('\n');

  const userPrompt = `ターン${turnNum}/${maxTurns} 耐心${patience}/100 前回TTFB:${lastTtfbMs ?? '-'}ms
${latestDefects?.length ? `前回問題:${latestDefects.join(';')}\n` : ''}未検証観点:
${checklist || '全て検証済み（自然に終話）'}
AI最新発言: ${latestAiText || '（応答待ち）'}`;

  const recentHistory = history.slice(-4).map(h => ({
    role: h.role === 'ai' ? 'user' : 'model',
    parts: [{ text: String(h.text || '（無言）').slice(0, 180) }]
  }));
  const contents = [
    ...recentHistory,
    { role: 'user', parts: [{ text: userPrompt }] }
  ];

  try {
    const { model, result } = await callGeminiJson({
      systemPrompt, contents, temperature: 0.75, maxOutputTokens: 160,
      disableThinking: true, timeoutMs: 6500, signal
    });
    const cleaned = sanitizeSpokenText(result.utterance_ja || '');
    const matched = objectives.find(o => o.turn === result.target_id) || fallback;
    return {
      model,
      utterance_ja: cleaned || (result.should_end ? '' : (matched?.text || '')),
      target_id: result.target_id || matched?.turn || turnNum,
      target_name: result.target_name || matched?.name || '动态追问',
      triggers: matched?.triggers || [],
      mode: result.mode || matched?.mode || 'normal',
      should_end: Boolean(result.should_end),
      covered_now: Array.isArray(result.covered_ids) ? result.covered_ids : (matched ? [matched.turn] : []),
      patience_delta: Number(result.patience_delta || 0),
      reason: result.reason || ''
    };
  } catch {
    return {
      model: 'fallback',
      utterance_ja: fallback?.text || '',
      target_id: fallback?.turn || turnNum,
      target_name: fallback?.name || '参考话术',
      triggers: fallback?.triggers || [],
      mode: fallback?.mode || 'normal',
      should_end: remaining.length === 0,
      covered_now: fallback ? [fallback.turn] : [],
      patience_delta: 0
    };
  }
}
