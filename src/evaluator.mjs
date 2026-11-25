import fs from 'fs';
import path from 'path';
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';

// 按技术根因去重后的单一问题标签映射（按实测高频排序，每个按钮对应 1 个独立技术子模块）
export const ISSUE_TAG_SPECS = [
  { tag: '响应过慢', module: 'Pipeline 链路延迟', cause: '疑似全链路首包延迟过高（ASR 端判 + LLM 首字 + TTS 流式合成耗时）' },
  { tag: 'ASR听错', module: 'ASR 语音识别', cause: '疑似 ASR 声学/语言模型识别偏差（同音词、人名、数字或日期误识）' },
  { tag: '读音错误', module: 'TTS / G2P 发音词典', cause: '疑似 TTS G2P 词典或日语汉字/量词读音规则错误' },
  { tag: '打断异常', module: 'VAD / 端点检测', cause: '疑似 VAD 静音阈值或打断判定策略异常（抢话重叠或误打断）' },
  { tag: '回复冗长', module: 'LLM / Prompt 约束', cause: '疑似 LLM 输出长度失控或 Prompt 口语精简约束不足' },
  { tag: '答非所问', module: 'LLM / 上下文理解', cause: '疑似 LLM 意图理解偏差或上下文状态丢失' },
  { tag: '棒读机械', module: 'TTS 韵律模型', cause: '疑似 TTS 日语高低音调/韵律停顿建模不足' },
  { tag: '敬语不当', module: 'LLM / 日语话术', cause: '疑似 LLM 日语敬语等级或商务礼仪 Prompt 约束不足' },
  { tag: '吞字截断', module: '音频流 / 分包缓冲', cause: '疑似音频流首尾截断、抖动缓冲欠载或端判过早切尾' },
  { tag: '事实幻觉', module: 'LLM / RAG 知识库', cause: '疑似 LLM 事实编造或知识库/业务规则检索未命中' },
  { tag: '音色突变', module: 'TTS 声码器一致性', cause: '疑似 TTS 跨句音色/风格一致性漂移或声码器失稳' },
  { tag: '读成中文', module: 'TTS 语种路由', cause: '疑似 TTS 多语种音素路由错误（日语汉字误切中文发音）' },
  { tag: '杂音爆音', module: '音频编码 / 增益削波', cause: '疑似声码器伪影、音频增益削波或编解码底噪' },
  { tag: '流程卡死', module: 'Dialog 状态机编排', cause: '疑似对话流程节点跳转异常或重复追问死循环' }
];

export const ISSUE_TAG_PRESETS = ISSUE_TAG_SPECS.map(s => s.tag);
export const ISSUE_TAG_MAP = Object.fromEntries(ISSUE_TAG_SPECS.map(s => [s.tag, s]));

// 根据人工勾选标签与客观指标，生成去重后的「疑似技术问题」列表
export function inferSuspectedTechIssues(tags = [], objectiveFlags = []) {
  const out = [];
  const seen = new Set();
  for (const tg of tags || []) {
    const spec = ISSUE_TAG_MAP[tg];
    const text = spec ? `[${spec.module}] ${spec.cause}` : `[待归因] ${tg}`;
    if (!seen.has(text)) { seen.add(text); out.push(text); }
  }
  for (const fl of objectiveFlags || []) {
    let text = '';
    if (fl.includes('首包延迟')) text = `[Pipeline 链路延迟] 客观监测触发：${fl}`;
    else if (fl.includes('削波')) text = `[音频编码 / 增益削波] 客观监测触发：${fl}`;
    else if (fl.includes('过静或缺失')) text = `[音频流 / 分包缓冲] 客观监测触发：${fl}`;
    if (text && !seen.has(text)) { seen.add(text); out.push(text); }
  }
  return out;
}

const FFMPEG_BIN = ['/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg'].find(p => fs.existsSync(p)) || 'ffmpeg';

export function analyzeAudioMetrics(wavPath) {
  const out = { exists: false, sizeBytes: 0, maxVolumeDb: null, meanVolumeDb: null, clipping: false, tooQuiet: false };
  if (!wavPath || !fs.existsSync(wavPath)) return out;
  const stat = fs.statSync(wavPath);
  out.exists = true;
  out.sizeBytes = stat.size;
  if (stat.size < 1200) { out.tooQuiet = true; return out; }

  try {
    const res = spawnSync(FFMPEG_BIN, ['-i', wavPath, '-af', 'volumedetect', '-f', 'null', '-'], {
      encoding: 'utf8', stdio: ['ignore', 'ignore', 'pipe']
    });
    const stderr = res.stderr || '';
    const maxM = stderr.match(/max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB/);
    const meanM = stderr.match(/mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB/);
    if (maxM) out.maxVolumeDb = parseFloat(maxM[1]);
    if (meanM) out.meanVolumeDb = parseFloat(meanM[1]);
    if (out.maxVolumeDb !== null && out.maxVolumeDb >= -0.5) out.clipping = true;
    if (out.meanVolumeDb !== null && out.meanVolumeDb <= -52) out.tooQuiet = true;
  } catch {}
  return out;
}

// 本地客观指标 + 人工评分槽位（保留已有人工评分，避免轮次结束覆盖通话中已打的分）
export function buildObjectiveTurnEval({ turn = 1, ttfbMs = null, aiAudioPath = null, existingReview = null }) {
  const audioMetrics = analyzeAudioMetrics(aiAudioPath);
  const objectiveFlags = [];
  if (turn !== 0 && ttfbMs != null && ttfbMs > 2500) objectiveFlags.push(`首包延迟偏高 (${ttfbMs}ms)`);
  if (aiAudioPath && (!audioMetrics.exists || audioMetrics.tooQuiet)) objectiveFlags.push('音频过静或缺失');
  else if (audioMetrics.clipping) objectiveFlags.push(`存在削波 (${audioMetrics.maxVolumeDb}dB)`);

  return {
    audio_metrics: audioMetrics,
    objectiveFlags,
    human_review: existingReview || {
      score: null,      // 1-5
      tags: [],         // 已选单一问题标签数组
      note: '',
      updated_at: null
    }
  };
}

function csvEscape(v) {
  const s = String(v ?? '').replace(/\r?\n/g, ' ');
  return /[",]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function buildRunReport(runRoot) {
  const scDirs = fs.readdirSync(runRoot)
    .map(name => path.join(runRoot, name))
    .filter(p => fs.statSync(p).isDirectory() && fs.existsSync(path.join(p, 'report_raw.json')));

  const scenariosSummary = [];
  const csvRows = [[
    '场景', '轮次', '发话方', '评测要点', '人工评分(1-5)', '问题标签', '疑似技术问题', '人工备注',
    '首包延迟(ms)', '峰值电平(dB)', '模拟用户发送文本', '客服ASR实识文本', '被测助手回复文本',
    '用户音频文件', '客服音频文件', '完整录屏文件'
  ]];

  for (const scDir of scDirs) {
    const raw = JSON.parse(fs.readFileSync(path.join(scDir, 'report_raw.json'), 'utf8'));
    const allTurns = [];
    if (raw.greeting) {
      allTurns.push({
        turn: 0,
        speaker_mode: 'ai',
        name: '开场白',
        ttfbMs: raw.greeting.ttfbMs ?? null,
        user_text: '',
        user_asr_text: '',
        ai_text: raw.greeting.text_delta || '',
        user_audio_file: null,
        ai_audio_file: raw.greeting.audio_file || null,
        eval: raw.greeting.eval || buildObjectiveTurnEval({ turn: 0, ttfbMs: raw.greeting.ttfbMs, aiAudioPath: raw.greeting.audio_file })
      });
    }
    for (const t of raw.turns || []) {
      allTurns.push({
        turn: t.turn,
        speaker_mode: t.speaker_mode || 'ai',
        name: t.name || '',
        ttfbMs: t.ai_metrics?.ttfbMs ?? null,
        user_text: t.user_text || '',
        user_asr_text: t.user_asr_text || '',
        ai_text: t.ai_text_delta || '',
        user_audio_file: t.user_audio_file || null,
        ai_audio_file: t.ai_audio_file || null,
        eval: t.eval || buildObjectiveTurnEval({ turn: t.turn, ttfbMs: t.ai_metrics?.ttfbMs, aiAudioPath: t.ai_audio_file })
      });
    }

    const scored = [];
    let reviewedCount = 0;
    let issueCount = 0;
    const tagCounter = {};
    const moduleCounter = {};

    for (const tr of allTurns) {
      const hr = tr.eval?.human_review || { score: null, tags: [], note: '' };
      if (typeof hr.score === 'number') {
        scored.push(hr.score);
        reviewedCount++;
      }
      const tags = Array.isArray(hr.tags) ? hr.tags : [];
      const techIssues = inferSuspectedTechIssues(tags, tr.eval?.objectiveFlags || []);
      tr.suspected_tech_issues = techIssues;

      if (tags.length > 0 || (typeof hr.score === 'number' && hr.score <= 3)) issueCount++;
      for (const tg of tags) {
        tagCounter[tg] = (tagCounter[tg] || 0) + 1;
        const mod = ISSUE_TAG_MAP[tg]?.module || '待归因';
        moduleCounter[mod] = (moduleCounter[mod] || 0) + 1;
      }

      const relUserWav = tr.user_audio_file ? `${raw.scenario_id}/${path.basename(tr.user_audio_file)}` : '';
      const relAiWav = tr.ai_audio_file ? `${raw.scenario_id}/${path.basename(tr.ai_audio_file)}` : '';
      const relVideo = raw.session_video ? `${raw.scenario_id}/${path.basename(raw.session_video)}` : '';

      csvRows.push([
        `${raw.agent_name} (${raw.scenario_id})`,
        `第 ${tr.turn} 轮`,
        tr.speaker_mode === 'human' ? '人工接管' : '模拟用户',
        tr.name,
        hr.score ?? '未评分',
        tags.join('；'),
        techIssues.join('；'),
        hr.note || '',
        tr.ttfbMs ?? '',
        tr.eval?.audio_metrics?.maxVolumeDb ?? '',
        tr.user_text,
        tr.user_asr_text,
        tr.ai_text,
        relUserWav,
        relAiWav,
        relVideo
      ]);
    }

    const avg5 = scored.length ? Number((scored.reduce((a, b) => a + b, 0) / scored.length).toFixed(2)) : null;
    const summary = {
      scenario_id: raw.scenario_id,
      agent_name: raw.agent_name,
      run_note: raw.run_note || '',
      total_turns: allTurns.length,
      reviewed_count: reviewedCount,
      pending_count: allTurns.length - reviewedCount,
      issue_count: issueCount,
      avg5,
      tag_counts: tagCounter,
      tech_module_counts: moduleCounter,
      session_video: raw.session_video || null,
      full_ai_audio: raw.full_ai_audio || null,
      turns: allTurns
    };
    fs.writeFileSync(path.join(scDir, 'evaluation.json'), JSON.stringify(summary, null, 2));
    scenariosSummary.push(summary);
  }

  const csvPath = path.join(runRoot, 'scorecard.csv');
  fs.writeFileSync(csvPath, '\uFEFF' + csvRows.map(r => r.map(csvEscape).join(',')).join('\n'), 'utf8');

  const md = [
    `# 语音智能体（Voice Agent）人工质检评测报告`,
    ``,
    `- 生成时间：${new Date().toISOString()}`,
    `- 数据文件：结构化真源 \`<场景>/report_raw.json\` 与 \`<场景>/evaluation.json\`；表格明细 \`scorecard.csv\``,
    ``,
    `## 一、场景人工评分总览`,
    ``,
    `| 场景 | 总轮次 | 已评 / 待评 | 人工均分（5分制） | 问题轮次 | 整场客服音频 | 完整录屏 |`,
    `| :--- | :---: | :---: | :---: | :---: | :--- | :--- |`
  ];
  for (const s of scenariosSummary) {
    const fullWav = s.full_ai_audio ? `\`${s.scenario_id}/${path.basename(s.full_ai_audio)}\`` : '未生成';
    const vid = s.session_video ? `\`${s.scenario_id}/${path.basename(s.session_video)}\`` : '未导出';
    md.push(`| ${s.agent_name} (${s.scenario_id}) | ${s.total_turns} | ${s.reviewed_count} / ${s.pending_count} | **${s.avg5 ?? '待评分'}** | ${s.issue_count} | ${fullWav} | ${vid} |`);
  }

  md.push(``, `## 二、高频问题与疑似技术模块归因`, ``);
  const globalTags = {};
  for (const s of scenariosSummary) {
    for (const [k, v] of Object.entries(s.tag_counts || {})) globalTags[k] = (globalTags[k] || 0) + v;
  }
  const sortedTags = Object.entries(globalTags).sort((a, b) => b[1] - a[1]);
  if (sortedTags.length) {
    for (const [k, v] of sortedTags) {
      const spec = ISSUE_TAG_MAP[k];
      const tech = spec ? ` → \`${spec.module}\`：${spec.cause}` : '';
      md.push(`- **${k}**（${v} 次）${tech}`);
    }
  } else {
    md.push(`- 暂未标记缺陷标签。`);
  }

  const notesWithText = scenariosSummary.filter(s => s.run_note && s.run_note.trim());
  if (notesWithText.length) {
    md.push(``, `## 三、人工总评与补充记录`, ``);
    for (const s of notesWithText) {
      md.push(`- **${s.agent_name} (${s.scenario_id})**：${s.run_note.trim().replace(/\r?\n/g, ' ')}`);
    }
  }

  md.push(``, `## 四、逐轮对话、ASR 实识、人工评分与疑似技术问题`, ``);
  md.push(`| 场景 | 轮次 | 评分 | 问题标签与备注 | 疑似技术问题 | 首包(ms) | 模拟用户原文 / 客服 ASR 实识 | 客服回复 | 对应音频文件 |`);
  md.push(`| :--- | :---: | :---: | :--- | :--- | :---: | :--- | :--- | :--- |`);
  for (const s of scenariosSummary) {
    for (const tr of s.turns) {
      const hr = tr.eval?.human_review || {};
      const scoreTxt = hr.score != null ? `**${hr.score} / 5**` : '待评';
      const issueTxt = [...(hr.tags || []), hr.note].filter(Boolean).join('<br>') || '-';
      const techTxt = (tr.suspected_tech_issues || []).join('<br>') || '-';
      const userCell = tr.turn === 0
        ? '（开场白）'
        : `<b>${tr.speaker_mode === 'human' ? '人工' : '模拟'}：</b>${(tr.user_text || '-').replace(/\n/g, ' ')}<br><b>ASR：</b>${(tr.user_asr_text || '未捕获').replace(/\n/g, ' ')}`;
      const aiCell = (tr.ai_text || '-').replace(/\n/g, ' ').slice(0, 120);
      const audioParts = [];
      if (tr.user_audio_file) audioParts.push(`用户：\`${s.scenario_id}/${path.basename(tr.user_audio_file)}\``);
      if (tr.ai_audio_file) audioParts.push(`客服：\`${s.scenario_id}/${path.basename(tr.ai_audio_file)}\``);
      const audioCell = audioParts.join('<br>') || '-';
      md.push(`| ${s.scenario_id} | 第 ${tr.turn} 轮 | ${scoreTxt} | ${issueTxt} | ${techTxt} | ${tr.ttfbMs ?? '-'} | ${userCell} | ${aiCell} | ${audioCell} |`);
    }
  }

  const mdPath = path.join(runRoot, 'scorecard_report.md');
  fs.writeFileSync(mdPath, md.join('\n'), 'utf8');
  return { mdPath, csvPath, scenariosSummary };
}

export function saveHumanReview(runRoot, scenarioId, turnNum, { score, tags, note } = {}) {
  const rawPath = path.join(runRoot, scenarioId, 'report_raw.json');
  if (!fs.existsSync(rawPath)) throw new Error('未找到对应的通话记录');
  const raw = JSON.parse(fs.readFileSync(rawPath, 'utf8'));

  let targetObj;
  if (turnNum === 0) {
    if (!raw.greeting) raw.greeting = { ttfbMs: null, text_delta: '', audio_file: null };
    targetObj = raw.greeting;
  } else {
    if (!Array.isArray(raw.turns)) raw.turns = [];
    targetObj = raw.turns.find(x => x.turn === turnNum);
    if (!targetObj) {
      targetObj = { turn: turnNum, speaker_mode: 'ai', name: `第 ${turnNum} 轮（进行中）`, user_text: '', user_asr_text: '', ai_text_delta: '' };
      raw.turns.push(targetObj);
      raw.turns.sort((a, b) => a.turn - b.turn);
    }
  }
  if (!targetObj.eval) {
    targetObj.eval = buildObjectiveTurnEval({
      turn: turnNum,
      ttfbMs: turnNum === 0 ? raw.greeting?.ttfbMs : targetObj.ai_metrics?.ttfbMs,
      aiAudioPath: turnNum === 0 ? raw.greeting?.audio_file : targetObj.ai_audio_file
    });
  }
  const prev = targetObj.eval.human_review || { score: null, tags: [], note: '' };
  const nextTags = Array.isArray(tags) ? tags : (prev.tags || []);
  targetObj.eval.human_review = {
    score: score === undefined ? prev.score : (score === null ? null : Math.max(1, Math.min(5, Number(score)))),
    tags: nextTags,
    note: typeof note === 'string' ? note : (prev.note || ''),
    suspected_tech_issues: inferSuspectedTechIssues(nextTags, targetObj.eval.objectiveFlags || []),
    updated_at: new Date().toISOString()
  };

  fs.writeFileSync(rawPath, JSON.stringify(raw, null, 2));
  return buildRunReport(runRoot);
}

// 保存本场底部人工总评/补充备注，并同步刷新 MD/CSV 报告
export function saveRunNote(runRoot, scenarioId, note) {
  const rawPath = path.join(runRoot, scenarioId, 'report_raw.json');
  if (!fs.existsSync(rawPath)) throw new Error('未找到对应的通话记录');
  const raw = JSON.parse(fs.readFileSync(rawPath, 'utf8'));
  raw.run_note = String(note ?? '');
  fs.writeFileSync(rawPath, JSON.stringify(raw, null, 2));
  return buildRunReport(runRoot);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const dir = process.argv[2];
  if (!dir) { console.error('用法: node src/evaluator.mjs <results目录>'); process.exit(1); }
  const res = buildRunReport(path.resolve(dir));
  console.log('报告已更新:', res.mdPath);
}
