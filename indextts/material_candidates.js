// Parameter-only postprocessing of cached CPU analysis. Keep parity with speech_intervals.
function candidateRanges(analysis, silenceSeconds, minimumVolumeDb, minimumSeconds) {
  const step = .032, duration = analysis.duration;
  const floor = minimumVolumeDb === null ? null : 10 ** (minimumVolumeDb / 20);
  const probabilities = analysis.probabilities.map((p, i) => floor === null || analysis.rms[i] >= floor ? p : 0);
  const raw = [];
  let start = null, silence = null;
  for (let i = 0; i < probabilities.length; i++) {
    const t = i * step, p = probabilities[i];
    if (start === null) { if (p >= .5) start = t; }
    else if (p < .35) {
      if (silence === null) silence = t;
      if (t + step - silence >= silenceSeconds - 1e-9) {
        if (silence - start >= .25) raw.push([start, Math.min(silence, duration)]);
        start = silence = null;
      }
    } else silence = null;
  }
  if (start !== null) {
    const end = Math.min(silence === null ? duration : silence, duration);
    if (end - start >= .25) raw.push([start, end]);
  }
  const clips = [];
  for (const [a, b] of raw) {
    let left = Math.max(0, a - .15), right = Math.min(duration, b + .15);
    if (clips.length && left < clips[clips.length - 1][1]) {
      const middle = (left + clips[clips.length - 1][1]) / 2;
      clips[clips.length - 1][1] = middle;
      left = middle;
    }
    while (right - left > 15) {
      const low = Math.trunc((left + 12) / step), high = Math.min(Math.trunc((left + 14.7) / step), probabilities.length - 1);
      let index = low;
      for (let j = low + 1; j <= high; j++) if (probabilities[j] < probabilities[index]) index = j;
      const cut = Math.min(left + 15, Math.max(left + 1, index * step));
      clips.push([left, cut]); left = cut;
    }
    clips.push([left, right]);
  }
  const round = value => Math.round(value * 1e6) / 1e6;
  const all = clips.map(([a, b], i) => [`c${i + 1}`, round(a), round(b), true]);
  return {rows: all.filter(r => r[2] - r[1] >= minimumSeconds - 1e-6), total: all.length,
    speechSeconds: raw.reduce((sum, [a, b]) => sum + b - a, 0)};
}

function stageRange(rows, editing, duration) {
  const [id, start, end] = editing;
  if (!Number.isFinite(start + end) || !(0 <= start && start < end && end <= duration))
    throw new Error('起点必须小于终点，且在素材范围内');
  if (end - start > 15 + 1e-6) throw new Error('每段最多 15 秒，请缩短范围');
  if (rows.some(r => r[3] && r[0] !== id && start < r[2] - 1e-6 && end > r[1] + 1e-6))
    throw new Error('与已暂存片段重叠，请调整边界或重新调整对应暂存片段');
  const result = structuredClone(rows), target = result.find(r => r[0] === id);
  if (target) target.splice(0, 4, id, start, end, true);
  else result.push([id, start, end, true]);
  return result.sort((a, b) => a[1] - b[1]);
}

if (typeof module !== 'undefined' && module.exports) module.exports = {candidateRanges, stageRange};
