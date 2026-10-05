"""Self-contained graphical source editor, without model or CDN dependencies."""
from html import escape
import json
from pathlib import Path
from urllib.parse import quote
import uuid


def player_html(path=None, *, record=None, values=None, primary="auto", peaks=None, analysis=None, workspace_id=None, editor_state=None):
    style = """<style>
    #material-waveform-draft{display:none!important}
    .material-wave{background:#f7f8fc;border:1px solid #e2e5ee;border-radius:16px;padding:18px;color:#34364a}
    .material-wave .tools{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:10px}
    .material-wave button{border:1px solid #d8d9eb;border-radius:7px;background:white;padding:5px 9px;font-size:13px;cursor:pointer;color:#34364a}
    .material-wave button:disabled{opacity:.4;cursor:default}
    .material-wave button:focus-visible,.material-wave svg:focus-visible{outline:2px solid #695aff;outline-offset:2px}
    .material-wave .clock{font-variant-numeric:tabular-nums;font-size:12px;margin-left:auto}
    .material-wave .editor-wave{width:100%;height:160px;background:white;border-radius:10px;touch-action:none;user-select:none;cursor:ew-resize;display:block}
    .material-wave .info{font-size:13px;min-height:22px;margin-top:6px}
    .material-wave .hint{font-size:12px;color:#74758b;margin-top:3px}
    .material-wave .audition-info{font-size:12px;color:#287a9d;margin-top:5px;font-variant-numeric:tabular-nums}
    .material-wave .navigation{margin-top:8px;padding:8px;background:#f7f8fc;border:1px solid #e6e8f0;border-radius:9px}
    .material-wave .navigation-tools{display:flex;align-items:center;gap:5px;flex-wrap:wrap;font-size:12px}
    .material-wave .navigation-tools span{margin-right:auto;color:#74758b}
    .material-wave .navigation-tools button{font-size:12px;padding:5px 7px}
    .material-wave .pan{width:100%;height:64px;display:block;touch-action:none;user-select:none;cursor:grab;margin-top:5px}
    .material-wave .pan[hidden]{display:none}
    .material-wave .pan:active{cursor:grabbing}
    .material-wave .navigation-hint{font-size:11px;color:#858699}
    .material-wave .workspace{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}
    .material-wave .pane{min-width:0;background:white;border:1px solid #e1e4ef;border-radius:12px;padding:12px}
    .material-wave .clip-list{max-height:410px;overflow:auto;display:grid;gap:8px;margin:10px 0 0;padding:2px}
    .material-wave .staged{max-height:230px}
    .material-wave .clip-row{border:1px solid #e6e8f0;border-radius:10px;padding:9px 10px;background:#fbfcfe}
    .material-wave .clip-row.is-selected{border-color:#d3ccff;background:#f3f0ff}
    .material-wave .clip-row.is-active{border-color:#8272eb;box-shadow:0 0 0 1px #8272eb;background:#f3f0ff}
    .material-wave .staged .clip-row{cursor:pointer}
    .material-wave .staged .clip-row:hover{border-color:#a195e7;background:#eeebff}
    .material-wave .staged .clip-row:focus-visible{outline:2px solid #8272eb;outline-offset:-2px}
    .material-wave .clip-row.is-disabled{border-color:#d9dde6;background:#f3f4f7;color:#73758b}
    .material-wave .clip-row.is-disabled.is-active{border-color:#9ba5b7;box-shadow:0 0 0 1px #9ba5b7}
    .material-wave .clip-enabled{display:flex;align-items:center;gap:4px;font-size:12px;cursor:pointer;white-space:nowrap}
    .material-wave .clip-enabled input{width:14px;height:14px;margin:0;appearance:auto;background-image:none;accent-color:#6b5ce7;cursor:pointer}
    .material-wave .clip-top,.material-wave .clip-bottom{display:flex;gap:8px;align-items:center;justify-content:space-between}
    .material-wave .clip-label{font-size:12px;font-variant-numeric:tabular-nums;color:#73758b}
    .material-wave .clip-title{font-size:13px;font-weight:600}
    .material-wave .clip-duration{font-size:12px;background:#eceef5;border-radius:5px;padding:1px 6px}
    .material-wave .clip-actions{display:flex;gap:5px}
    .material-wave .clip-wave{width:100%;height:32px;display:block;margin:3px 0;pointer-events:none}
    .material-wave .add-clip{background:#6b5ce7;color:white;border-color:#6b5ce7}
    .material-wave .selected-clip{color:#6b5ce7;border-color:#d3ccff;background:#ece8ff}
    .material-wave .source-overview{width:100%;height:85px;background:white;border-radius:10px;display:block;margin:8px 0 16px}
    .material-wave .legend{font-size:12px;color:#777a91;display:flex;gap:15px;align-items:center}
    .material-wave .legend .dot{width:8px;height:8px;background:#aab1c4;border-radius:3px;display:inline-block;margin-right:5px}
    .material-wave .legend .selected-dot{background:#7c69e6}
    .material-wave .pane-header{display:flex;align-items:center;justify-content:space-between;gap:10px}
    .material-wave h4{font-size:15px;margin:12px 0 6px}
    .material-wave .editor-panel{margin-top:12px;padding-top:12px;border-top:1px solid #e6e8f1}
    .material-wave .editor-panel[hidden]{display:none}
    .material-wave .edit-controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:8px;background:#f3f0ff;border-radius:8px;margin-top:8px}
    .material-wave .edit-controls input{width:105px;border:1px solid #ddd;border-radius:6px;background:white;padding:6px;color:#34364a}
    .material-wave .candidate-info,.material-wave .staged-info,.material-wave .edit-info{font-size:13px;min-height:22px}
    .material-wave .edit-info{color:#79738f;font-size:12px;margin-top:5px}
    .material-wave .empty{padding:24px 12px;text-align:center;color:#878ba1;border:1px dashed #d9ddeb;border-radius:10px;font-size:13px}
    .material-wave .empty-icon{font-size:26px;color:#a195e7;margin-bottom:8px}
    .material-wave details{font-size:12px;color:#777a91;margin-top:8px}
    @media(max-width:760px){.material-wave .workspace{grid-template-columns:1fr}.material-wave{padding:10px}.material-wave .candidates{max-height:280px}}
    </style>"""
    if path is None:
        return style + '<div class="material-wave empty"><div class="empty-icon">♫</div>上传音频后，这里会显示波形和备选片段。</div>'
    url = "/gradio_api/file=" + quote(str(path).replace("\\", "/"), safe="/:")
    data = {"sourceId": record["sourceId"], "duration": record["durationSeconds"],
            "speech": record["speechSeconds"], "rows": values, "primary": primary, "peaks": peaks,
            "analysis": analysis, "revision": record.get('revision'),
            "candidates": [[s['id'], s['start'], s['end'], True] for s in record['segments']]}
    if workspace_id:
        data.update(workspaceId=workspace_id, editorState=editor_state or {})
    payload = escape(json.dumps(data, ensure_ascii=False, separators=(",", ":")), quote=True)
    # A command may intentionally restore the same backend value after browser-only edits.
    return style + f'''<div id="material-waveform" class="material-wave" data-render="{uuid.uuid4().hex}" data-config="{payload}">
      <audio id="producer-material-audio" preload="metadata" src="{escape(url, quote=True)}"></audio>
      <div class="pane-header"><strong>音频总览</strong><div class="legend"><span><i class="dot"></i>备选</span><span><i class="dot selected-dot"></i>已选</span><span>{record['durationSeconds']/60:.1f} 分钟</span></div></div>
      <svg class="source-overview" role="img" aria-label="完整音频与备选、已选片段分布" viewBox="0 0 1000 85"></svg>
      <div class="workspace">
      <section class="pane candidate-pane" aria-label="备选片段区">
        <h4>② 备选片段</h4><div class="candidate-info" role="status" aria-live="polite"></div>
        <div class="clip-list candidates" role="list" aria-label="实时候选分段"></div>
      </section>
      <section class="pane selected-pane" aria-label="选中片段区">
        <div class="pane-header"><h4>③ 选中片段</h4><button type="button" data-action="undo">↶ 撤销</button></div>
        <div class="staged-info" role="status" aria-live="polite"></div>
        <div class="clip-list staged" role="list" aria-label="选中片段"></div>
      <div class="editor-panel" hidden>
      <div class="pane-header"><strong class="editor-title">微调当前片段</strong><span class="editor-duration"></span></div>
      <div class="tools">
        <button type="button" data-action="play" aria-label="播放或暂停音轨" hidden>▶ 播放</button>
        <button type="button" data-action="preview" aria-label="从指定起点试听或暂停">▶ 从起点试听</button>
        <button type="button" data-action="restart" aria-label="从当前片段开头播放">↶ 从头播放</button>
        <button type="button" data-action="out" aria-label="缩小音轨">−</button>
        <button type="button" data-action="in" aria-label="放大音轨">＋</button>
        <span class="clock" aria-live="off"></span>
      </div>
      <svg class="editor-wave" role="application" aria-label="选中片段波形：点击指定试听起点，拖动左右把手微调" tabindex="0" viewBox="0 0 1000 190"></svg>
      <div class="audition-info" aria-live="polite"></div>
      <div class="navigation">
        <div class="navigation-tools"><span>局部导航</span>
          <button type="button" data-action="pan-left" aria-label="视野向前移动 0.5 秒">← 0.5 秒</button>
          <button type="button" data-action="fit" aria-label="视野回到当前片段">回到片段</button>
          <button type="button" data-action="pan-right" aria-label="视野向后移动 0.5 秒">0.5 秒 →</button>
        </div>
        <svg class="pan" viewBox="0 0 1000 140" role="slider" tabindex="0" aria-label="局部音轨视野" aria-describedby="material-navigation-hint"></svg>
        <div class="navigation-hint" id="material-navigation-hint">拖动紫色框移动视野 · 方向键精移 0.1 秒，Shift 加方向键 0.01 秒</div>
      </div>
      <div class="info" role="status" aria-live="polite"></div>
      <div class="edit-info" role="status" aria-live="polite">拖动左右把手微调，松手自动保留。</div>
      <details><summary>精确起止时间</summary><div class="edit-controls">
        <label>起点（秒） <input type="number" data-bound="start" step="0.001" min="0" aria-label="当前片段起点（秒）" disabled></label>
        <label>终点（秒） <input type="number" data-bound="end" step="0.001" min="0" aria-label="当前片段终点（秒）" disabled></label>
      </div></details>
      </div>
      <details><summary>更多操作</summary><button type="button" data-action="clear">清空选中列表</button></details>
      </section></div>
    </div>'''


INITIALIZE = Path(__file__).with_name("material_waveform.js").read_text(encoding="utf-8").replace(
    '/* CANDIDATE_ALGORITHM */', Path(__file__).with_name('material_candidates.js').read_text(encoding='utf-8'))
