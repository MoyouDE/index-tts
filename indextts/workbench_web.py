"""Gradio controls for persistent voices and local reader audition."""
import functools
import json
from pathlib import Path

import gradio as gr

from .runtime.profiles import FP32, BF16, InferenceOptimizations, generation_options

PROFILES = [("FP32", FP32), ("BF16", BF16)]
PARAMETERS = ["do_sample", "temperature", "top_p", "top_k", "num_beams", "repetition_penalty",
              "length_penalty", "max_generate_length", "acoustic_steps", "cfg"]


def ui_errors(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc
    return wrapped


def choices(service):
    return [(f"{item['displayName']} · {item['voiceId']}", item["voiceId"]) for item in service.library.items()]


def library_controls(service, session, source, device):
    gr.Markdown("## 本地音色库\n每种精度只保留最新包和最近一次试听。更换参考音频请用新的音色 ID。")
    gr.Textbox(value=str(service.library.root), label="本地工作区（不进入 Git）", interactive=False)
    with gr.Row():
        selected = gr.Dropdown(choices(service), label="库中音色", interactive=True)
        profile = gr.Radio(PROFILES, value=FP32, label="库中版本")
        refresh = gr.Button("刷新音色库")
    version_status = gr.Textbox(label="版本状态", interactive=False)
    with gr.Accordion("音色包元数据", open=False):
        details = gr.JSON(label="音色信息与版本状态")
    reference = gr.Audio(label="库存参考音频", interactive=False)
    package = gr.File(label="下载所选版本", interactive=False)
    notes = gr.Textbox(label="本地备注", lines=2)
    with gr.Row():
        save = gr.Button("保存备注")
        rebuild = gr.Button("生成／替换所选精度", variant="primary")
    status = gr.Textbox(label="音色库操作结果", interactive=False)
    with gr.Accordion("导入已有音色包", open=False):
        imported = gr.File(label="导入 .ivp（无参考音频）", file_types=[".ivp"])
        import_button = gr.Button("导入本地库")
    with gr.Accordion("删除本地音色", open=False):
        deletion = gr.Textbox(label="将删除的音色及范围", interactive=False)
        confirm = gr.Checkbox(label="确认删除此音色的参考音频、两档包及试听结果", value=False)
        delete = gr.Button("删除选中音色", variant="stop")
        token = gr.State(None)

    @ui_errors
    def show(voice, precision):
        if not voice:
            return {}, None, None, "", "", None, False, ""
        record = service.library.record(voice)
        item = next(item for item in service.library.items() if item["voiceId"] == voice)
        audio = None
        if record["reference"]:
            audio = (service.library.directory(voice) / record["reference"]).resolve()
            if audio.parent != service.library.directory(voice):
                raise ValueError("参考音频路径无效")
        pack = service.library.pack_path(voice, precision)
        states = " · ".join(f"{label}: {'已生成' if item['variants'][p].get('ready') else item['variants'][p].get('error', '尚未生成')}" for label, p in PROFILES)
        return (item, str(audio) if audio else None, str(pack) if pack.exists() else None, record["notes"],
                f"{record['displayName']}（{voice}）：参考音频、所有精度包、试听及备注", f"{voice}:{record['instance']}", False, states)

    outputs = [details, reference, package, notes, deletion, token, confirm, version_status]
    selected.change(show, [selected, profile], outputs)
    profile.change(show, [selected, profile], outputs)
    refresh.click(lambda: gr.update(choices=choices(service), value=None), outputs=selected)
    save.click(ui_errors(lambda voice, note: (service.library.notes(voice, note), "备注已保存")[1]), [selected, notes], status)

    @ui_errors
    def regenerate(voice, precision, dev, models, sid, progress=gr.Progress()):
        if not voice:
            raise ValueError("请先选择库中音色")
        path, result = service.rebuild(voice, precision, dev, models, sid, progress)
        return path, f"{precision} 生成并校验完成：{result['seconds']:.2f} 秒"

    rebuild.click(regenerate, [selected, profile, device, source, session], [package, status], concurrency_limit=None).then(show, [selected, profile], outputs)

    @ui_errors
    def import_pack(path):
        if not path:
            raise ValueError("请上传 .ivp")
        installed = service.library.install(path)
        return gr.update(choices=choices(service), value=Path(installed).parent.name), "音色包已导入"

    import_button.click(import_pack, imported, [selected, status])

    @ui_errors
    def remove(voice, target, approved):
        if not approved:
            raise ValueError("请勾选确认，并核对将删除的音色")
        service.library.delete(voice, target)
        return gr.update(choices=choices(service), value=None), "音色及本地产物已删除"

    delete.click(remove, [selected, token, confirm], [selected, status])
    return selected, profile


def audition_controls(service, session, emotion_result, tabs, library_voice, library_profile):
    saved = service.library.settings()
    with gr.Tab("合成试听", id="audition") as audition_tab:
        gr.Markdown("选择本地音色包与同精度的裁剪模型。模型目录需自行准备；首次加载和切换配置可能较慢。")
        with gr.Row():
            voice = gr.Dropdown(choices(service), label="试听音色", interactive=True)
            precision = gr.Radio(PROFILES, value=FP32, label="试听精度")
            refresh = gr.Button("刷新试听音色")
        ready = gr.Textbox(label="音色版本状态", interactive=False)
        missing = gr.Button("前往制包页生成所选精度")
        model = gr.Textbox(label="已有裁剪模型目录", value=saved.get("models", {}).get(FP32, ""))
        device = gr.Dropdown(["cuda:0", "cuda:1"], value=saved.get("device", "cuda:0"), allow_custom_value=True, label="试听 CUDA 设备")
        text = gr.Textbox(label="试听文本", value="清晨的阳光照进了安静的书房。", lines=3)
        with gr.Row():
            emotion_mode = gr.Radio(["基础情感", "手动八维向量", "采用情感页结果"], value="基础情感", label="试听情感")
            vector = gr.Textbox(label="八维向量（JSON）", value="[0,0,0,0,0,0,0,0]")
        gr.Markdown("八维顺序：高兴、愤怒、悲伤、恐惧、厌恶、低落、惊讶、平静。采用情感页结果时使用本会话最近一次成功分析的向量；实际有效情感随试听报告保存。")
        with gr.Row():
            speed = gr.Slider(0.5, 2, value=1, step=0.05, label="语速倍率（越大越快）")
            random_seed = gr.Checkbox(value=False, label="每次随机种子")
            seed = gr.Number(value=17, precision=0, label="固定种子", minimum=0, maximum=2**32-1)
        with gr.Accordion("高级推理设置", open=False):
            reset = gr.Button("恢复当前精度默认参数")
            with gr.Row():
                sample = gr.Checkbox(value=False, label="随机采样 do_sample")
                temperature = gr.Slider(0.1, 2, value=0.8, step=0.1, label="temperature")
                top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="top_p")
                top_k = gr.Slider(0, 100, value=30, step=1, label="top_k（0 表示不限制）")
            with gr.Row():
                beams = gr.Slider(1, 10, value=3, step=1, label="beam 数")
                repetition = gr.Slider(0.1, 20, value=10, step=0.1, label="重复惩罚")
                length = gr.Slider(-2, 2, value=0, step=0.1, label="长度惩罚")
                tokens = gr.Number(value=1500, precision=0, minimum=50, label="最大生成 token（受模型容量限制）")
            with gr.Row():
                steps = gr.Slider(1, 100, value=25, step=1, label="声学步数")
                cfg = gr.Slider(0, 3, value=0.7, step=0.05, label="CFG")
            optimizations = gr.Radio(["全部启用", "全部关闭", "自定义"], value="全部启用", label="推理优化")
            custom = gr.CheckboxGroup(list(InferenceOptimizations().to_dict()), value=list(InferenceOptimizations().to_dict()), label="自定义优化项")
            with gr.Row():
                threads = gr.Number(value=saved.get("cpuThreads", 4), precision=0, minimum=1, maximum=64, label="CPU 线程")
                allocator = gr.Radio(["native", "cudaMallocAsync"], value=saved.get("allocator", "native"), label="显存分配器")
        controls = [sample, temperature, top_p, top_k, beams, repetition, length, tokens, steps, cfg]
        def defaults(profile):
            settings = {"temperature": 0.8, "top_p": 0.8, "top_k": 30, "acoustic_steps": 25, "cfg": 0.7, **generation_options(profile)}
            return [settings[name] for name in PARAMETERS]
        reset.click(defaults, precision, controls)
        precision.change(defaults, precision, controls)
        with gr.Row():
            run = gr.Button("生成试听", variant="primary")
            cancel = gr.Button("取消当前会话试听", variant="stop")
            unload = gr.Button("卸载试听模型")
        status = gr.Textbox(label="试听状态", interactive=False)
        audio = gr.Audio(label="最近一次试听", interactive=False, type="filepath")
        with gr.Row():
            download = gr.File(label="下载试听 WAV", interactive=False)
            report_file = gr.File(label="下载试听配置 JSON", interactive=False)
        with gr.Accordion("试听参数、耗时与显存统计", open=False):
            report = gr.JSON(label="试听报告")

        @ui_errors
        def select(vid, profile):
            current = service.library.settings().get("models", {}).get(profile, "")
            if not vid:
                return "请选择音色", None, None, None, {}, current
            path = service.library.pack_path(vid, profile)
            wav, detail, data = service.library.latest(vid, profile)
            return ("已生成，可选择匹配的模型试听" if path.exists() else "尚未生成：请前往制包页生成此精度", wav, wav, detail, data, current)
        selected_outputs = [ready, audio, download, report_file, report, model]
        voice.change(select, [voice, precision], selected_outputs)
        precision.change(select, [voice, precision], selected_outputs)
        def remember_model(profile, value):
            service.library.save_settings({"models": {**service.library.settings().get("models", {}), profile: value}})
        model.blur(ui_errors(remember_model), [precision, model])
        def refresh_voices(current):
            available = choices(service)
            return gr.update(choices=available, value=current if current in {v for _, v in available} else None)
        refresh.click(refresh_voices, voice, voice).then(select, [voice, precision], selected_outputs)
        audition_tab.select(refresh_voices, voice, voice).then(select, [voice, precision], selected_outputs)
        missing.click(lambda v, p: (gr.update(selected="voices"), gr.update(choices=choices(service), value=v), p),
                      [voice, precision], [tabs, library_voice, library_profile])

        @ui_errors
        def synth(vid, profile, models, dev, txt, mode, manual, analyzed, rate, randomize, fixed,
                  opts, names, cpu, alloc, sid, sampling, temp, p, k, beam, rep, penalty, limit, count, guidance,
                  progress=gr.Progress()):
            if not vid:
                raise ValueError("请先选择试听音色")
            emotion = "base"
            if mode == "手动八维向量":
                emotion = json.loads(manual)
            elif mode == "采用情感页结果":
                if not analyzed:
                    raise ValueError("请先在情感页完成一次分析")
                emotion = analyzed["vector"]
            optimization = "all" if opts == "全部启用" else "none" if opts == "全部关闭" else ",".join(names) or "none"
            values = [sampling, temp, p, k, beam, rep, penalty, limit, count, guidance]
            try:
                wav, details, result = service.audition(vid, profile, models, dev, txt, emotion, rate,
                    None if randomize else fixed, dict(zip(PARAMETERS, values)), optimization, cpu, alloc, sid, progress)
            except RuntimeError as exc:
                if "取消" not in str(exc):
                    raise
                return gr.skip(), gr.skip(), gr.skip(), gr.skip(), "试听已取消，上一次成功结果仍保留"
            info = result["result"]
            message = f"完成 · 模型加载 {result['modelLoadMs']/1000:.2f}s · 合成 {info['timings']['totalMs']/1000:.2f}s · 音频 {info['durationMs']/1000:.2f}s · RTF {info['rtf']:.3f} · 种子 {result['seed']}"
            return wav, wav, details, result, message
        run.click(synth, [voice, precision, model, device, text, emotion_mode, vector, emotion_result,
                         speed, random_seed, seed, optimizations, custom, threads, allocator, session, *controls],
                  [audio, download, report_file, report, status], concurrency_limit=None)
        cancel.click(service.cancel, session, status, queue=False)
        unload.click(ui_errors(service.unload_audition), outputs=status, concurrency_limit=None)
        return voice
