"""Self-contained graphical source editor, without model or CDN dependencies."""
from html import escape
import json
from pathlib import Path
from urllib.parse import quote
import uuid


def player_html(path=None, *, record=None, values=None, primary="auto", peaks=None):
    style = """<style>
    #material-waveform-draft{display:none!important}
    .material-wave{background:#f8f9ff;border:1px solid #dedff0;border-radius:12px;padding:12px;color:#34364a}
    .material-wave .tools{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:10px}
    .material-wave button{border:1px solid #d8d9eb;border-radius:7px;background:white;padding:5px 9px;font-size:13px;cursor:pointer;color:#34364a}
    .material-wave button:disabled{opacity:.4;cursor:default}
    .material-wave button:focus-visible,.material-wave svg:focus-visible{outline:2px solid #695aff;outline-offset:2px}
    .material-wave .clock{font-variant-numeric:tabular-nums;font-size:12px;margin-left:auto}
    .material-wave svg{width:100%;height:190px;background:white;border-radius:8px;touch-action:none;user-select:none;cursor:crosshair;display:block}
    .material-wave .info{font-size:13px;min-height:22px;margin-top:6px}
    .material-wave .hint{font-size:12px;color:#74758b;margin-top:3px}
    .material-wave .pan{width:100%;margin:6px 0 0;accent-color:#695aff}
    </style>"""
    if path is None:
        return style + '<div class="material-wave">上传素材后，直接在完整波形上拖选多个片段。</div>'
    url = "/gradio_api/file=" + quote(str(path).replace("\\", "/"), safe="/:")
    data = {"sourceId": record["sourceId"], "duration": record["durationSeconds"],
            "speech": record["speechSeconds"], "rows": values, "primary": primary, "peaks": peaks}
    payload = escape(json.dumps(data, ensure_ascii=False, separators=(",", ":")), quote=True)
    # A command may intentionally restore the same backend value after browser-only edits.
    return style + f'''<div id="material-waveform" class="material-wave" data-render="{uuid.uuid4().hex}" data-config="{payload}">
      <audio id="producer-material-audio" preload="metadata" src="{escape(url, quote=True)}"></audio>
      <div class="tools">
        <button type="button" data-action="play" aria-label="播放或暂停音轨">▶ 播放</button>
        <button type="button" data-action="preview">试听选段</button>
        <button type="button" data-action="primary">设为主参考</button>
        <button type="button" data-action="delete">删除选段</button>
        <button type="button" data-action="undo">撤销</button>
        <button type="button" data-action="out" aria-label="缩小音轨">−</button>
        <button type="button" data-action="in" aria-label="放大音轨">＋</button>
        <span class="clock" aria-live="off"></span>
      </div>
      <svg role="application" aria-label="素材波形：拖动空白新增片段，拖动边界调整" tabindex="0" viewBox="0 0 1000 190"></svg>
      <input class="pan" type="range" min="0" max="1" step="0.001" value="0" aria-label="横向移动音轨" hidden>
      <div class="info" role="status" aria-live="polite"></div>
      <div class="hint">检测语音 {record['speechSeconds']:.2f} 秒 · 空白处拖选，点击选中，拖动两端调整 · 双击试听 · 每段 ≤15 秒</div>
    </div>'''


INITIALIZE = Path(__file__).with_name("material_waveform.js").read_text(encoding="utf-8")
