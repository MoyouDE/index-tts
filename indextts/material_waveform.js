() => {
  requestAnimationFrame(() => requestAnimationFrame(() => {
    const root = document.getElementById('material-waveform');
    if (!root || root._editor) return;
    root._editor = true;
    /* CANDIDATE_ALGORITHM */
    const config = JSON.parse(root.dataset.config);
    const svg = root.querySelector('.editor-wave'), overview = root.querySelector('.source-overview'), audio = root.querySelector('audio');
    const info = root.querySelector('.info'), clock = root.querySelector('.clock');
    const editInfo = root.querySelector('.edit-info'), pan = root.querySelector('.pan');
    const ns = 'http://www.w3.org/2000/svg', duration = config.duration;
    const storageKey = `material-review-v2:${config.workspaceId || 'legacy'}:${config.sourceId}`;
    let rows = structuredClone(config.rows), primary = config.primary, editing = null, previewStarts = {};
    let disabled = new Set(config.editorState?.disabled || []);
    let zoom = 1, offset = 0, drag = null, stop = null, frame = null, history = [], candidates = [], playingButton = null;
    let navigationStart = 0, navigationEnd = duration, navigationDrag = null;
    if (config.editorState) {
      previewStarts = config.editorState.previewStarts || {};
      editing = config.editorState.editing || null;
    }
    try {
      const saved = JSON.parse(sessionStorage.getItem(storageKey));
      if (!config.workspaceId && saved && saved.revision === config.revision && Array.isArray(saved.rows)) {
        rows = saved.rows; primary = saved.primary;
        editing = saved.editing && typeof saved.editing[1] === 'number' && typeof saved.editing[2] === 'number'
          && Number.isFinite(saved.editing[1] + saved.editing[2]) ? saved.editing : null;
        if (saved.previewStarts && typeof saved.previewStarts === 'object' && !Array.isArray(saved.previewStarts))
          previewStarts = Object.fromEntries(Object.entries(saved.previewStarts).filter(([, t]) => typeof t === 'number' && Number.isFinite(t)));
      }
    } catch (_) { /* Storage can be disabled; the current page still works. */ }
    const clamp = (t, a = 0, b = duration) => Math.min(b, Math.max(a, t));
    const fmt = t => Number.isFinite(t) ? `${Math.floor(t / 60)}:${(t % 60).toFixed(2).padStart(5, '0')}` : '—';
    const x = t => (t - offset) / (duration / zoom) * 1000;
    const at = event => {
      const box = svg.getBoundingClientRect();
      return clamp(Math.round((offset + (event.clientX - box.left) / box.width * duration / zoom) * 1000) / 1000);
    };
    const chosen = () => rows.filter(r => r[3]).sort((a, b) => a[1] - b[1]);
    const enabledChosen = () => chosen().filter(r => !disabled.has(r[0]));
    const button = name => root.querySelector(`[data-action="${name}"]`);
    const bound = side => root.querySelector(`[data-bound="${side}"]`);
    const previewStart = range => clamp(previewStarts[range[0]] ?? range[1], range[1], Math.max(range[1], range[2] - .001));
    function setPreviewStart(time) {
      if (!editing || !Number.isFinite(time) || !Number.isFinite(editing[1] + editing[2]) || editing[2] <= editing[1]) return;
      previewStarts[editing[0]] = clamp(time, editing[1], Math.max(editing[1], editing[2] - .001));
      audio.currentTime = previewStarts[editing[0]];
    }
    const effectivePrimary = () => enabledChosen().find(r => r[0] === primary)?.[0] ||
      enabledChosen().reduce((best, r) => !best || r[2] - r[1] > best[2] - best[1] ? r : best, null)?.[0];
    const scale = 58 / Math.max(.05, ...config.peaks.low.map(Math.abs), ...config.peaks.high.map(Math.abs));
    function remember() {
      try { sessionStorage.setItem(storageKey, JSON.stringify({revision: config.revision, rows, primary,
        editing: editing && Number.isFinite(editing[1] + editing[2]) ? editing : null, previewStarts})); } catch (_) {}
      if (config.workspaceId) {
        root.dataset.editorState = JSON.stringify({workspaceId:config.workspaceId, sourceId:config.sourceId,
          rows, primary, disabled:[...disabled], editing, previewStarts, zoom, offset});
        document.dispatchEvent(new Event('producer-editor-change'));
      }
    }
    function publish() {
      remember();
      const field = document.querySelector('#material-waveform-draft textarea');
      if (!field) return;
      const value = JSON.stringify({sourceId: config.sourceId, rows, primary,
        ...(config.workspaceId ? {workspaceId:config.workspaceId, disabled:[...disabled], editing, previewStarts, zoom, offset} : {})});
      Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(field, value);
      field.dispatchEvent(new Event('input', {bubbles: true}));
    }
    function checkpoint() {
      history.push({rows: structuredClone(rows), primary, disabled:[...disabled]});
      if (history.length > 30) history.shift();
    }
    function el(tag, attrs, parent = svg) {
      const node = document.createElementNS(ns, tag);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
      parent.appendChild(node); return node;
    }
    const linkedRange = r => chosen().find(s => r[1] < s[2] - 1e-6 && r[2] > s[1] + 1e-6);
    function thumbnail(range, color) {
      const mini = document.createElementNS(ns, 'svg'); mini.setAttribute('viewBox', '0 0 200 32'); mini.classList.add('clip-wave'); mini.setAttribute('aria-hidden', 'true');
      const {low, high} = config.peaks, values = [];
      for (let i = 0; i < 50; i++) {
        const index = Math.min(low.length - 1, Math.floor((range[1] + (range[2] - range[1]) * i / 50) / duration * low.length));
        values.push(Math.max(Math.abs(low[index]), Math.abs(high[index])));
      }
      const peak = Math.max(.0001, ...values);
      let d = '';
      values.forEach((v, i) => { const height = v / peak * 13; d += `M${i * 4 + 2},${16 - height}V${16 + height}`; });
      el('path', {d, stroke: color, 'stroke-width': 2, 'stroke-linecap': 'round'}, mini);
      return mini;
    }
    function renderList(container, list, staged) {
      container.replaceChildren();
      if (!list.length) {
        const empty = document.createElement('div'); empty.className = 'empty';
        empty.textContent = staged ? '点击备选卡片的“＋ 加入”，在这里微调和试听。' : '暂无备选片段，试着降低音量或时长门槛。';
        container.appendChild(empty); return;
      }
      list.forEach((r, i) => {
        const linked = staged ? r : linkedRange(r);
        const off = staged && disabled.has(r[0]);
        const item = document.createElement('div'); item.className = 'clip-row' + (linked ? ' is-selected' : '') + (off ? ' is-disabled' : '') + (staged && r[0] === editing?.[0] ? ' is-active' : ''); item.setAttribute('role', 'listitem');
        if (staged) {
          item.dataset.editClipId = r[0]; item.tabIndex = 0;
          item.setAttribute('aria-label', `微调选中片段 ${i + 1}`);
          item.title = '点击卡片微调';
        }
        const top = document.createElement('div'); top.className = 'clip-top';
        const title = document.createElement('span'); title.className = 'clip-title'; title.textContent = `${staged ? '选中' : '备选'} ${i + 1}${staged && r[0] === effectivePrimary() ? ' ★' : ''}`;
        const seconds = document.createElement('span'); seconds.className = 'clip-duration'; seconds.textContent = `${(r[2] - r[1]).toFixed(2)} 秒`;
        top.append(title, seconds); item.append(top, thumbnail(r, linked && !disabled.has(linked[0]) ? '#8874dd' : '#9ba8c5'));
        if (staged && config.workspaceId) {
          const label = document.createElement('label'); label.className = 'clip-enabled';
          label.title = off ? '停用：不参与音色制作，片段仍保留' : '启用：参与音色制作';
          const toggle = document.createElement('input'); toggle.type = 'checkbox'; toggle.checked = !off;
          toggle.dataset.clipToggle = r[0]; toggle.setAttribute('aria-label', `启用选中片段 ${i + 1}`);
          label.append(toggle, document.createTextNode('启用')); top.insertBefore(label, seconds);
        }
        const bottom = document.createElement('div'); bottom.className = 'clip-bottom';
        const label = document.createElement('span'); label.className = 'clip-label'; label.textContent = `${fmt(r[1])}–${fmt(r[2])}`;
        const actions = document.createElement('div'); actions.className = 'clip-actions'; bottom.append(label, actions);
        for (const [action, text] of (staged ? [['listen', '▶'], ['remove', '×']] : [['listen', '▶'], ['add', linked ? '已选 ✓' : '＋ 加入']])) {
          const control = document.createElement('button'); control.type = 'button'; control.textContent = text;
          control.dataset.clipAction = action; control.dataset.clipId = r[0]; control.dataset.clipKind = staged ? 'staged' : 'candidate';
          control.dataset.idleText = text;
          const verb = {listen: '试听', remove: '移除', add: linked ? '定位已选' : '加入'}[action];
          control.setAttribute('aria-label', `${verb}${staged ? '选中' : '备选'}片段 ${i + 1}`);
          control.title = action === 'add' && linked ? '此范围已有选中片段，点击查看' : verb;
          if (action === 'add') control.className = linked ? 'selected-clip' : 'add-clip';
          actions.appendChild(control);
        }
        item.append(bottom);
        if (staged) {
          const main = document.createElement('button'); main.type = 'button'; main.textContent = r[0] === effectivePrimary() ? '★' : '☆';
          main.title = '设为主参考'; main.setAttribute('aria-label', `设主参考选中片段 ${i + 1}`);
          main.dataset.clipAction = 'main'; main.dataset.clipId = r[0]; main.dataset.clipKind = 'staged';
          main.disabled = off;
          top.insertBefore(main, seconds);
        }
        container.appendChild(item);
      });
    }
    function renderStaged() {
      const list = chosen(), enabled = enabledChosen();
      root.querySelector('.staged-info').textContent = config.workspaceId
        ? `已选 ${list.length} 段 · 启用 ${enabled.length} 段 / ${enabled.reduce((s,r)=>s+r[2]-r[1],0).toFixed(2)} 秒${list.length>enabled.length ? ` · 停用 ${list.length-enabled.length} 段` : ''}`
        : `${list.length} 段已选 · 共 ${list.reduce((s, r) => s + r[2] - r[1], 0).toFixed(2)} 秒 · 点击卡片微调`;
      renderList(root.querySelector('.staged'), list, true);
      button('clear').disabled = !list.length;
    }
    function controls() {
      const value = (id, fallback) => Number(document.querySelector(`#${id} input[type="number"]`)?.value || fallback);
      return {silence: value('material-silence', .5), minimum: value('material-minimum', 1),
        volume: value('material-volume', -50), enabled: !!document.querySelector('#material-volume-enabled input')?.checked};
    }
    let filters = controls();
    function updateCandidates(event) {
      if (event) {
        const control = event.target.closest('#material-silence,#material-minimum,#material-volume,#material-volume-enabled');
        if (!control) return;
        const key = {'material-silence': 'silence', 'material-minimum': 'minimum', 'material-volume': 'volume', 'material-volume-enabled': 'enabled'}[control.id];
        filters[key] = key === 'enabled' ? event.target.checked : Number(event.target.value);
      }
      if (!Number.isFinite(filters.silence + filters.minimum + filters.volume) || filters.silence < .1 || filters.silence > 3 || filters.minimum < .1 || filters.minimum > 15 || filters.volume < -80 || filters.volume > 0) return;
      const result = config.analysis ? candidateRanges(config.analysis, filters.silence, filters.enabled ? filters.volume : null, filters.minimum) :
        {rows: config.candidates.filter(r => r[2] - r[1] >= filters.minimum - 1e-6), total: config.candidates.length, speechSeconds: config.speech};
      candidates = result.rows;
      root.querySelector('.candidate-info').textContent = `${candidates.length} 段备选 · 已隐藏 ${result.total - candidates.length} 段短声音${config.analysis ? '' : ' · 此旧素材仅支持时长筛选'}`;
      renderList(root.querySelector('.candidates'), candidates, false);
      draw();
    }
    // Gradio can replace slider/checkbox DOM nodes. One delegated listener follows the active source.
    if (document._materialFilterListener) {
      document.removeEventListener('input', document._materialFilterListener);
      document.removeEventListener('change', document._materialFilterListener);
    }
    document._materialFilterListener = event => { if (root.isConnected) updateCandidates(event); };
    document.addEventListener('input', document._materialFilterListener);
    document.addEventListener('change', document._materialFilterListener);
    function updateNavigation() {
      const span = duration / zoom;
      const context = Math.max(5, editing ? editing[2] - editing[1] : 0, span);
      navigationStart = clamp(Math.min((editing?.[1] ?? offset) - context, offset));
      navigationEnd = clamp(Math.max((editing?.[2] ?? offset + span) + context, offset + span));
    }
    function fitEditingView() {
      if (!editing) return;
      const span = Math.min(duration, Math.max(3, (editing[2] - editing[1]) * 1.7));
      zoom = duration / span; offset = clamp((editing[1] + editing[2] - span) / 2, 0, duration - span);
      updateNavigation();
    }
    function focusEditing(range) {
      audio.pause();
      editing = [...range];
      audio.currentTime = previewStart(editing);
      fitEditingView();
      remember(); renderStaged(); draw();
      const selectedList = root.querySelector('.staged'), activeCard = selectedList.querySelector('.is-active');
      if (activeCard) {
        const cardBox = activeCard.getBoundingClientRect(), listBox = selectedList.getBoundingClientRect();
        if (cardBox.top < listBox.top || cardBox.bottom > listBox.bottom)
          selectedList.scrollTop += cardBox.top - listBox.top - 4;
      }
    }
    function addCandidate(range) {
      const linked = linkedRange(range);
      if (linked) { focusEditing(linked); return; }
      try {
        const next = ['s' + crypto.randomUUID().replaceAll('-', '').slice(0, 10), range[1], range[2], true];
        const updated = stageRange(rows, next, duration);
        checkpoint(); rows = updated; publish(); focusEditing(next);
        renderList(root.querySelector('.candidates'), candidates, false); draw();
      } catch (error) { editInfo.textContent = error.message; }
    }
    function applyEditing() {
      if (!editing) return;
      try {
        const updated = stageRange(rows, editing, duration);
        if (Number.isFinite(previewStarts[editing[0]])) previewStarts[editing[0]] = previewStart(editing);
        const previous = rows.find(r => r[0] === editing[0]);
        if (!previous || previous[1] !== editing[1] || previous[2] !== editing[2]) {
          checkpoint(); rows = updated; publish(); renderStaged(); renderList(root.querySelector('.candidates'), candidates, false);
        }
        updateNavigation(); draw(); editInfo.textContent = '微调已保留 · 可继续试听，或点击撤销恢复。';
      } catch (error) {
        const original = rows.find(r => r[3] && r[0] === editing[0]); editing = original ? [...original] : null;
        draw(); editInfo.textContent = error.message;
        if (editing) { bound('start').value = editing[1]; bound('end').value = editing[2]; }
      }
    }
    function drawOverview() {
      overview.replaceChildren();
      const {low, high} = config.peaks;
      let d = '';
      for (let i = 0; i < low.length; i += Math.max(1, Math.floor(low.length / 700))) {
        const height = Math.max(Math.abs(low[i]), Math.abs(high[i])) * scale * .45;
        d += `M${i / low.length * 1000},${35 - height}V${35 + height}`;
      }
      el('path', {d, stroke: '#b5bed0', 'stroke-width': 1}, overview);
      for (const [list, kind, color] of [[candidates, 'candidate', '#aab1c4'], [chosen(), 'staged', '#8975dd']]) {
        for (const r of list) {
          const off = kind === 'staged' && disabled.has(r[0]);
          const box = el('rect', {x: r[1] / duration * 1000, y: 9, width: Math.max(2, (r[2] - r[1]) / duration * 1000), height: 53, fill: off ? '#adb5c6' : color, 'fill-opacity': off ? .25 : .55, rx: 2, 'data-id': r[0], 'data-kind': kind, style: 'cursor:pointer'}, overview);
          const title = el('title', {}, box); title.textContent = `${kind === 'staged' ? (off ? '已选（停用）' : '已选') : '备选'} ${fmt(r[1])}–${fmt(r[2])}`;
        }
      }
      for (let i = 0; i <= 4; i++) { const tick = el('text', {x: i * 250, y: 79, fill: '#9a9db0', 'font-size': 12, 'text-anchor': i === 4 ? 'end' : 'start'}, overview); tick.textContent = fmt(duration * i / 4); }
    }
    function drawNavigation() {
      pan.replaceChildren(); pan.hidden = zoom <= 1;
      const span = duration / zoom, width = Math.max(.001, navigationEnd - navigationStart);
      const px = t => (t - navigationStart) / width * 1000;
      const {low, high} = config.peaks;
      const first = Math.max(0, Math.floor(navigationStart / duration * low.length));
      const last = Math.min(low.length, Math.ceil(navigationEnd / duration * low.length));
      const scale = 35 / Math.max(.0001, ...low.slice(first, last).map(Math.abs), ...high.slice(first, last).map(Math.abs));
      let d = '';
      el('rect', {x: 0, y: 6, width: 1000, height: 88, rx: 12, fill: '#e9ecf4'}, pan);
      if (editing) el('rect', {x: clamp(px(editing[1]), 0, 1000), y: 6,
        width: Math.max(1, Math.min(1000, px(editing[2])) - Math.max(0, px(editing[1]))), height: 88, fill: '#c5b9f5', 'fill-opacity': .7}, pan);
      for (let i = first; i < last; i += Math.max(1, Math.floor((last - first) / 180))) {
        const h = Math.max(Math.abs(low[i]), Math.abs(high[i])) * scale;
        d += `M${px(i / low.length * duration)},${50 - h}V${50 + h}`;
      }
      el('path', {d, stroke: '#a2a9c2', 'stroke-width': 2, 'pointer-events': 'none'}, pan);
      const left = px(offset), viewWidth = span / width * 1000;
      el('rect', {x: left, y: 6, width: Math.max(2, viewWidth), height: 88, rx: 12, fill: '#7c69e6', 'fill-opacity': .12,
        stroke: '#7c69e6', 'stroke-width': 3, 'data-window': '', style: 'cursor:grab'}, pan);
      el('rect', {x: left - Math.max(0, 70 - viewWidth) / 2, y: 6, width: Math.max(70, viewWidth), height: 88,
        fill: 'transparent', 'data-window': '', style: 'cursor:grab'}, pan);
      for (const [t, anchor] of [[navigationStart, 'start'], [navigationEnd, 'end']]) {
        const tick = el('text', {x: px(t), y: 132, 'text-anchor': anchor, fill: '#77788e', 'font-size': 26, 'pointer-events': 'none'}, pan); tick.textContent = fmt(t);
      }
      pan.setAttribute('aria-valuemin', navigationStart);
      pan.setAttribute('aria-valuemax', Math.max(navigationStart, navigationEnd - span));
      pan.setAttribute('aria-valuenow', offset);
      pan.setAttribute('aria-valuetext', `当前显示 ${fmt(offset)} 至 ${fmt(offset + span)}`);
      button('pan-left').disabled = offset <= .00001;
      button('pan-right').disabled = offset + span >= duration - .00001;
    }
    function draw() {
      drawOverview();
      root.querySelector('.editor-panel').hidden = !editing;
      svg.replaceChildren();
      const {low, high} = config.peaks, n = low.length;
      const begin = Math.max(0, Math.floor(offset / duration * n));
      const end = Math.min(n, Math.ceil((offset + duration / zoom) / duration * n));
      const localScale = 58 / Math.max(.0001, ...low.slice(begin, end).map(Math.abs), ...high.slice(begin, end).map(Math.abs));
      let d = '', top = [], bottom = [];
      const stride = Math.max(1, Math.floor((end - begin) / 1000));
      for (let i = begin; i < end; i += stride) {
        const px = x(i / n * duration), lo = Math.min(...low.slice(i, Math.min(end, i + stride))), hi = Math.max(...high.slice(i, Math.min(end, i + stride)));
        d += `M${px.toFixed(2)},${(102 - hi * localScale).toFixed(2)}V${(102 - lo * localScale).toFixed(2)}`;
        top.push(`${px.toFixed(2)},${(102 - hi * localScale).toFixed(2)}`);
        bottom.push(`${px.toFixed(2)},${(102 - lo * localScale).toFixed(2)}`);
      }
      if (top.length) el('path', {d: `M${top.join('L')}L${bottom.reverse().join('L')}Z`, fill: '#a79ce0', 'fill-opacity': .35, 'pointer-events': 'none'});
      el('path', {d, stroke: '#8581b4', 'stroke-width': 2.5, fill: 'none', 'pointer-events': 'none'});
      for (let i = 0; i <= 5; i++) {
        const t = offset + i * duration / zoom / 5, px = i * 200;
        el('line', {x1: px, x2: px, y1: 28, y2: 174, stroke: '#ebecf5', 'pointer-events': 'none'});
        const tick = el('text', {x: clamp(px, 2, 910), y: 186, fill: '#77788e', 'font-size': 18, 'pointer-events': 'none'}); tick.textContent = fmt(t);
      }
      const ranges = chosen().filter(r => r[0] !== editing?.[0]).map(r => [r, 'staged']);
      if (editing && Number.isFinite(editing[1] + editing[2])) ranges.push([editing, 'editing']);
      ranges.forEach(([r, kind]) => {
        if (x(r[2]) < 0 || x(r[1]) > 1000) return;
        const start = clamp(x(r[1]), 0, 1000), end = clamp(x(r[2]), 0, 1000), focused = kind === 'editing';
        const color = focused && !disabled.has(r[0]) ? '#7c69e6' : '#bbc1d3';
        const region = el('g', {'data-id': r[0], 'data-kind': kind, role: 'button', 'aria-label': `${focused ? '正在微调' : '已选'}片段，${r[1].toFixed(3)} 至 ${r[2].toFixed(3)} 秒`, tabindex: 0});
        el('rect', {x: start, y: 27, width: Math.max(1, end - start), height: 145, rx: 3, fill: color, 'fill-opacity': focused ? .25 : .12, stroke: color}, region);
        if (focused) for (const [side, t] of [['start', r[1]], ['end', r[2]]]) {
          if (x(t) >= 0 && x(t) <= 1000) {
            el('rect', {x: clamp(x(t) - 12, 0, 976), y: 47, width: 24, height: 110, rx: 8, fill: color, 'data-edge': side, style: 'cursor:ew-resize', role: 'slider', tabindex: 0,
              'aria-label': side === 'start' ? '调整片段起点' : '调整片段终点', 'aria-valuemin': 0, 'aria-valuemax': duration, 'aria-valuenow': t}, region);
            el('line', {x1: x(t), x2: x(t), y1: 88, y2: 118, stroke: 'white', 'stroke-width': 3, 'pointer-events': 'none'}, region);
          }
        }
      });
      if (editing && Number.isFinite(editing[1] + editing[2]) && editing[2] > editing[1]) {
        const start = previewStart(editing), px = x(start);
        if (px >= 0 && px <= 1000) {
          const marker = el('g', {'data-preview-start': '', role: 'slider', tabindex: 0, 'aria-label': '指定试听起点',
            'aria-valuemin': editing[1], 'aria-valuemax': Math.max(editing[1], editing[2] - .001),
            'aria-valuenow': start, 'aria-valuetext': fmt(start), style: 'cursor:ew-resize'});
          el('rect', {x: px - 14, y: 3, width: 28, height: 24, fill: 'transparent'}, marker);
          el('line', {x1: px, x2: px, y1: 13, y2: 175, stroke: '#2683a9', 'stroke-width': 2, 'stroke-dasharray': '5 4', 'pointer-events': 'none'}, marker);
          el('path', {d: `M${px - 9},3H${px + 9}L${px},16Z`, fill: '#2683a9', 'pointer-events': 'none'}, marker);
        }
      }
      el('line', {x1: x(audio.currentTime), x2: x(audio.currentTime), y1: 22, y2: 175, stroke: '#ee8657', 'stroke-width': 2, 'pointer-events': 'none', 'data-playhead': ''});
      root.querySelector('.audition-info').textContent = editing && Number.isFinite(editing[1] + editing[2]) && editing[2] > editing[1]
        ? `试听起点 ${fmt(previewStart(editing))} · 点击波形指定，蓝色标记可拖动` : '';
      info.textContent = editing ? `${fmt(editing[1])} → ${fmt(editing[2])}` : '';
      root.querySelector('.editor-title').textContent = editing ? `微调选中 ${chosen().findIndex(r => r[0] === editing[0]) + 1}${disabled.has(editing[0]) ? ' · 已停用' : ''}` : '微调片段';
      root.querySelector('.editor-duration').textContent = editing && Number.isFinite(editing[2] - editing[1]) ? `${(editing[2] - editing[1]).toFixed(2)} 秒` : '';
      for (const [side, index] of [['start', 1], ['end', 2]]) {
        const input = bound(side); input.disabled = !editing; input.max = duration;
        if (document.activeElement !== input) input.value = editing && Number.isFinite(editing[index]) ? Number(editing[index].toFixed(3)) : '';
      }
      let error = '';
      if (editing) try { stageRange(rows, editing, duration); } catch (e) { error = e.message; }
      button('preview').disabled = !editing || !Number.isFinite(editing[1] + editing[2]) || editing[1] < 0 || editing[2] <= editing[1] || editing[2] > duration;
      button('restart').disabled = button('preview').disabled;
      button('undo').disabled = !history.length;
      button('out').disabled = zoom <= 1; button('in').disabled = duration / zoom <= .25;
      drawNavigation();
      clock.textContent = `${fmt(audio.currentTime)} / ${fmt(duration)}`;
      editInfo.textContent = error || '拖动左右把手微调，松手自动保留 · 每段最多 15 秒';
    }
    function playbackTick() {
      if (!root.isConnected) { audio.pause(); return; }
      if (stop !== null && audio.currentTime >= stop) { const end = stop; audio.pause(); audio.currentTime = end; }
      const head = svg.querySelector('[data-playhead]');
      if (head) { head.setAttribute('x1', x(audio.currentTime)); head.setAttribute('x2', x(audio.currentTime)); }
      clock.textContent = `${fmt(audio.currentTime)} / ${fmt(duration)}`;
      if (!audio.paused) frame = requestAnimationFrame(playbackTick);
    }
    function play(start = null, end = null) {
      if (playingButton) { playingButton.textContent = playingButton.dataset.idleText || '▶'; playingButton.setAttribute('aria-pressed', 'false'); playingButton = null; }
      audio.pause(); if (start !== null) audio.currentTime = start; stop = end;
      audio.play().catch(e => { info.textContent = `无法播放：${e.message}`; stop = null; });
    }
    audio.addEventListener('play', () => { button('play').textContent = button('preview').textContent = '❚❚ 暂停'; cancelAnimationFrame(frame); playbackTick(); });
    audio.addEventListener('pause', () => {
      if (!audio.paused) return;
      button('play').textContent = '▶ 播放'; button('preview').textContent = '▶ 从起点试听'; stop = null; cancelAnimationFrame(frame);
      if (playingButton) { playingButton.textContent = playingButton.dataset.idleText || '▶'; playingButton.setAttribute('aria-pressed', 'false'); playingButton = null; }
    });
    audio.addEventListener('timeupdate', () => { if (audio.paused) playbackTick(); else if (stop !== null && audio.currentTime >= stop) { const end = stop; audio.pause(); audio.currentTime = end; } });
    audio.addEventListener('error', () => { info.textContent = '音轨播放失败，请检查素材文件是否可读取'; });
    function rangeFromRegion(region) {
      return region.dataset.kind === 'editing' ? editing : (region.dataset.kind === 'staged' ? rows : candidates).find(r => r[0] === region.dataset.id);
    }
    svg.addEventListener('pointerdown', event => {
      if (event.button !== 0) return;
      const region = event.target.closest('[data-id]');
      if (region && region.dataset.kind !== 'editing') {
        const r = rangeFromRegion(region); if (r) focusEditing(r); return;
      }
      if (!editing) return;
      audio.pause(); svg.focus(); svg.setPointerCapture(event.pointerId);
      drag = {time: at(event), pixel: event.clientX, before: [...editing], previewBefore: previewStarts[editing[0]],
        mode: event.target.closest('[data-preview-start]') ? 'seek' : region ? (event.target.dataset.edge || 'move') : 'seek', moved: false};
    });
    svg.addEventListener('pointermove', event => {
      if (!drag || (Math.abs(event.clientX - drag.pixel) < 3 && !drag.moved)) return;
      drag.moved = true; const t = at(event);
      if (drag.mode === 'seek') setPreviewStart(t);
      else if (editing) {
        if (drag.mode === 'start') editing[1] = t;
        else if (drag.mode === 'end') editing[2] = t;
        else { const delta = clamp(t - drag.time, -drag.before[1], duration - drag.before[2]); editing[1] = drag.before[1] + delta; editing[2] = drag.before[2] + delta; }
      }
      draw();
    });
    svg.addEventListener('pointerup', event => {
      if (!drag) return;
      if (!drag.moved && !['start', 'end'].includes(drag.mode)) setPreviewStart(drag.time);
      const moved = drag.moved && drag.mode !== 'seek';
      drag = null;
      if (moved) applyEditing(); else { remember(); draw(); }
      if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId);
    });
    svg.addEventListener('pointercancel', () => {
      if (drag) { editing = drag.before; previewStarts[editing[0]] = drag.previewBefore; audio.currentTime = previewStart(editing); drag = null; draw(); }
    });
    svg.addEventListener('dblclick', event => {
      if (!editing || event.target.closest('[data-edge]')) return;
      setPreviewStart(at(event)); remember(); draw(); play(previewStart(editing), editing[2]);
    });
    root.addEventListener('input', event => {
      const side = event.target.dataset.bound;
      if (!side || !editing) return;
      editing[side === 'start' ? 1 : 2] = event.target.valueAsNumber;
      draw();
    });
    root.addEventListener('change', event => {
      const id = event.target.dataset.clipToggle;
      if (id) {
        checkpoint();
        if (event.target.checked) disabled.delete(id); else disabled.add(id);
        publish(); renderStaged(); renderList(root.querySelector('.candidates'), candidates, false); draw();
        root.querySelector(`[data-clip-toggle="${id}"]`)?.focus({preventScroll:true});
      }
      if (event.target.dataset.bound) applyEditing();
    });
    overview.addEventListener('click', event => {
      const region = event.target.closest('[data-id]'); if (!region) return;
      const range = rangeFromRegion(region);
      if (range) { if (region.dataset.kind === 'candidate') addCandidate(range); else focusEditing(range); }
    });
    root.addEventListener('click', event => {
      if (event.target.closest('.clip-enabled')) return;
      const clip = event.target.closest('[data-clip-action]');
      if (clip) {
        const staged = clip.dataset.clipKind === 'staged';
        const r = (staged ? rows : candidates).find(r => r[0] === clip.dataset.clipId); if (!r) return;
        const action = clip.dataset.clipAction;
        if (action === 'listen') {
          if (playingButton === clip && !audio.paused) audio.pause();
          else { play(staged ? previewStart(r) : r[1], r[2]); playingButton = clip; clip.textContent = '❚❚'; clip.setAttribute('aria-pressed', 'true'); }
        }
        if (action === 'add') addCandidate(r);
        if (action === 'main') { checkpoint(); primary = r[0]; publish(); renderStaged(); draw(); }
        if (action === 'remove') {
          checkpoint(); r[3] = false; disabled.delete(r[0]); if (primary === r[0]) primary = 'auto';
          if (editing?.[0] === r[0]) editing = chosen()[0] ? [...chosen()[0]] : null;
          publish(); renderStaged(); renderList(root.querySelector('.candidates'), candidates, false);
          if (editing) focusEditing(editing); else draw();
        }
        return;
      }
      const card = event.target.closest('[data-edit-clip-id]');
      if (card) { activateCard(card); return; }
      const action = event.target.closest('[data-action]')?.dataset.action; if (!action) return;
      if (action === 'play') { if (audio.paused) play(); else audio.pause(); }
      if (action === 'preview' && editing) { if (audio.paused) play(previewStart(editing), editing[2]); else audio.pause(); }
      if (action === 'restart' && editing) play(editing[1], editing[2]);
      if (action === 'fit') { fitEditingView(); draw(); remember(); }
      if (action === 'pan-left' || action === 'pan-right') {
        offset = clamp(offset + (action === 'pan-right' ? .5 : -.5), 0, duration - duration / zoom);
        updateNavigation(); draw(); remember();
      }
      if (action === 'clear') { checkpoint(); rows = rows.map(r => [r[0], r[1], r[2], false]); disabled.clear(); primary = 'auto'; editing = null; publish(); renderStaged(); renderList(root.querySelector('.candidates'), candidates, false); draw(); }
      if (action === 'undo' && history.length) {
        const previous = history.pop(); rows = previous.rows; primary = previous.primary; disabled = new Set(previous.disabled);
        editing = chosen().find(r => r[0] === editing?.[0]) || chosen()[0] || null;
        if (editing) editing = [...editing];
        publish(); renderStaged(); renderList(root.querySelector('.candidates'), candidates, false);
        if (editing) focusEditing(editing); else draw();
      }
      if (action === 'in' || action === 'out') {
        const center = offset + duration / zoom / 2;
        zoom = clamp(zoom * (action === 'in' ? 2 : .5), 1, Math.max(1, duration / .25)); offset = clamp(center - duration / zoom / 2, 0, duration - duration / zoom); updateNavigation(); draw(); remember();
      }
    });
    function activateCard(card) {
      const range = rows.find(r => r[3] && r[0] === card.dataset.editClipId);
      if (!range) return;
      focusEditing(range);
      root.querySelector(`[data-edit-clip-id="${range[0]}"]`)?.focus({preventScroll: true});
    }
    root.addEventListener('keydown', event => {
      if (event.target.matches('[data-edit-clip-id]') && ['Enter', ' '].includes(event.key)) {
        event.preventDefault(); activateCard(event.target);
      }
    });
    svg.addEventListener('keydown', event => {
      if (editing && event.target.closest('[data-preview-start]') && ['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.code)) {
        event.preventDefault(); audio.pause();
        const t = event.code === 'Home' ? editing[1] : event.code === 'End' ? editing[2] - .001 :
          previewStart(editing) + (event.code === 'ArrowRight' ? 1 : -1) * (event.shiftKey ? .01 : .1);
        setPreviewStart(Math.round(t * 1000) / 1000); remember(); draw(); svg.querySelector('[data-preview-start]')?.focus(); return;
      }
      const side = event.target.dataset.edge;
      if (editing && side && ['ArrowLeft', 'ArrowRight'].includes(event.code)) {
        event.preventDefault(); editing[side === 'start' ? 1 : 2] = Math.round((editing[side === 'start' ? 1 : 2] + (event.code === 'ArrowRight' ? 1 : -1) * (event.shiftKey ? .01 : .1)) * 1000) / 1000;
        applyEditing(); svg.querySelector(`[data-edge="${side}"]`)?.focus();
      }
      if (event.code === 'Space') { event.preventDefault(); button('preview').click(); }
      if (event.code === 'Escape' && editing) { const saved = rows.find(r => r[0] === editing[0]); if (saved) focusEditing(saved); }
      if (event.code === 'Enter') { event.preventDefault(); applyEditing(); button('preview').click(); }
    });
    const navigationAt = event => {
      const box = pan.getBoundingClientRect();
      return navigationStart + (event.clientX - box.left) / box.width * (navigationEnd - navigationStart);
    };
    pan.addEventListener('pointerdown', event => {
      if (event.button !== 0) return;
      event.preventDefault(); pan.focus(); pan.setPointerCapture(event.pointerId);
      const time = navigationAt(event), before = offset;
      if (!event.target.closest('[data-window]')) offset = clamp(time - duration / zoom / 2, navigationStart, Math.max(navigationStart, navigationEnd - duration / zoom));
      navigationDrag = {time, offset, before}; draw();
    });
    pan.addEventListener('pointermove', event => {
      if (!navigationDrag) return;
      offset = clamp(navigationDrag.offset + navigationAt(event) - navigationDrag.time, navigationStart, Math.max(navigationStart, navigationEnd - duration / zoom)); draw();
    });
    pan.addEventListener('pointerup', event => {
      navigationDrag = null;
      if (pan.hasPointerCapture(event.pointerId)) pan.releasePointerCapture(event.pointerId);
      remember();
    });
    pan.addEventListener('pointercancel', () => { if (navigationDrag) { offset = navigationDrag.before; navigationDrag = null; draw(); } });
    pan.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.code)) return;
      event.preventDefault();
      const last = Math.max(navigationStart, navigationEnd - duration / zoom);
      offset = event.code === 'Home' ? navigationStart : event.code === 'End' ? last :
        clamp(Math.round((offset + (event.code === 'ArrowRight' ? 1 : -1) * (event.shiftKey ? .01 : .1)) * 1000) / 1000, navigationStart, last);
      draw(); remember();
    });
    editing = chosen().find(r => r[0] === editing?.[0]) || chosen().find(r => r[0] === primary) || chosen()[0] || null;
    renderStaged(); updateCandidates();
    if (editing) focusEditing(editing);
    if (config.editorState && Number.isFinite(config.editorState.zoom) && config.editorState.zoom >= 1
        && Number.isFinite(config.editorState.offset)) {
      zoom = config.editorState.zoom;
      offset = clamp(config.editorState.offset, 0, Math.max(0, duration-duration/zoom));
      updateNavigation(); draw();
    }
    publish();
  }));
  return [];
}
