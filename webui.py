import html
import json
import os
import re
import shutil
import sys
import threading
import time

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(
    description="IndexTTS WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--version", type=str, default="2.5", choices=["2", "2.5"], help="Model version to use")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
parser.add_argument("--accel", action="store_true", default=False, help="Use GPT2 acceleration engine if available")
parser.add_argument("--torch_compile", action="store_true", default=False, help="Use torch.compile to optimize s2mel if available")
parser.add_argument("--qwen_emo", action="store_true", default=False, help="Load QwenEmotion even on a low-VRAM GPU, where it is skipped by default")
parser.add_argument("--gui_seg_tokens", type=int, default=120, help="GUI: Max tokens per generation segment")
cmd_args = parser.parse_args()

# Validate optional acceleration dependencies early, so missing extras fail
# at startup instead of halfway through inference.
def _require_optional_extra(flag_name, module_name, install_cmd):
    try:
        __import__(module_name)
    except ImportError:
        parser.error(
            f"--{flag_name} requires {module_name}, which is not installed. "
            f"Install it with: {install_cmd}"
        )

if cmd_args.accel:
    _require_optional_extra("accel", "flash_attn", "uv sync --extra accel")

if cmd_args.torch_compile:
    _require_optional_extra("torch_compile", "triton", "uv sync --extra torch_compile")

REQUIRED_FILES = {
    "2": ["bpe.model", "gpt.pth", "s2mel.pth", "wav2vec2bert_stats.pt"],
    "2.5": [
        "gpt.pth",
        "s2mel.pth",
        "codec.pth",
        "multilingual_zh_ja_yue_char_del.tiktoken",
        "wav2vec2bert_stats.pt",
    ],
}
MODEL_REPO = {
    "2": "IndexTeam/IndexTTS-2",
    "2.5": "IndexTeam/IndexTTS-2.5",
}
required_files = REQUIRED_FILES[cmd_args.version]
missing = [f for f in required_files if not os.path.exists(os.path.join(cmd_args.model_dir, f))]
if missing:
    print(
        f"Model directory {cmd_args.model_dir} is incomplete for v{cmd_args.version} "
        f"(missing: {', '.join(missing)}). Downloading {MODEL_REPO[cmd_args.version]}..."
    )
    from indextts.utils.model_download import snapshot_download
    try:
        snapshot_download(MODEL_REPO[cmd_args.version], local_dir=cmd_args.model_dir)
    except Exception as e:
        print(f"Failed to download model to {cmd_args.model_dir}: {e}")
        sys.exit(1)
    missing = [f for f in required_files if not os.path.exists(os.path.join(cmd_args.model_dir, f))]
    if missing:
        print(f"Failed to download model to {cmd_args.model_dir} (still missing: {', '.join(missing)}). Please download it manually.")
        sys.exit(1)
    print("Model downloaded successfully.")

from indextts.utils.model_download import ensure_config_available
try:
    ensure_config_available(cmd_args.model_dir, version=cmd_args.version)
except Exception as e:
    print(f"Failed to download config.yaml: {e}")
    sys.exit(1)

IS_V25 = cmd_args.version == "2.5"

import gradio as gr
from indextts.utils.examples_downloader import ensure_examples_available
from indextts.utils.presets import list_presets, save_preset, load_preset, delete_preset
from tools.i18n.i18n import I18nAuto

if IS_V25:
    from indextts.infer_v2_5 import IndexTTS2
    from indextts.voicepack import VoicePackBuilder, load_voicepack
else:
    from indextts.infer_v2 import IndexTTS2

i18n = I18nAuto(language="Auto")
MODE = 'local'

# Download example audio files if missing
ensure_examples_available()

LOW_VRAM_THRESHOLD_GB = 10.0


def detect_vram_gb():
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


_vram_gb = detect_vram_gb()
LOW_VRAM = _vram_gb is not None and _vram_gb < LOW_VRAM_THRESHOLD_GB

HALF_PRECISION = cmd_args.fp16 or LOW_VRAM
LOAD_QWEN_EMO = cmd_args.qwen_emo or not LOW_VRAM

if LOW_VRAM:
    print(
        f">> {_vram_gb:.1f} GB VRAM detected (< {LOW_VRAM_THRESHOLD_GB:.0f} GB): "
        "enabling half precision"
        + ("" if LOAD_QWEN_EMO else " and skipping QwenEmotion")
    )
    if not LOAD_QWEN_EMO:
        print(">> Emotion control from a text description is unavailable; pass --qwen_emo to force it.")


def build_tts(use_accel=False, use_torch_compile=False):
    """Build an IndexTTS2 instance with the requested acceleration options."""
    import torch

    kwargs = dict(
        model_dir=cmd_args.model_dir,
        cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
        use_deepspeed=cmd_args.deepspeed,
        use_cuda_kernel=cmd_args.cuda_kernel,
        use_accel=use_accel,
        use_torch_compile=use_torch_compile,
        use_qwen_emo=LOAD_QWEN_EMO,
    )
    if IS_V25:
        use_bf16 = HALF_PRECISION and torch.cuda.is_bf16_supported()
        if HALF_PRECISION and not use_bf16:
            print(">> BF16 is not supported on this device, falling back to full precision.")
        kwargs["use_bf16"] = use_bf16
    else:
        kwargs["use_fp16"] = HALF_PRECISION
    return IndexTTS2(**kwargs)


tts = build_tts(use_accel=cmd_args.accel, use_torch_compile=cmd_args.torch_compile)
voicepack_builder = VoicePackBuilder(
    tts,
    model_dir=cmd_args.model_dir,
    cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
) if IS_V25 else None
VOICEPACK_DIR = os.path.abspath(os.path.join("outputs", "voicepacks"))
os.makedirs(VOICEPACK_DIR, exist_ok=True)
DEFAULT_VOICEPACK_EXPORT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "index-tts-package", "SoundPackage")
)
VOICEPACK_EXPORT_DIR = os.path.abspath(
    os.environ.get("INDEXTTS_VOICEPACK_EXPORT_DIR", DEFAULT_VOICEPACK_EXPORT_DIR)
)
os.makedirs(VOICEPACK_EXPORT_DIR, exist_ok=True)
_voicepack_cache = {}
mutex = threading.Lock()
# 支持的语言列表
LANGUAGES = {
    "中文": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES_ALL = [i18n("与音色参考音频相同"),
                i18n("使用情感参考音频"),
                i18n("使用情感向量控制"),
                i18n("使用情感描述文本控制")]
EMO_CHOICES_OFFICIAL = EMO_CHOICES_ALL[:-1]  # skip experimental features
EMO_CHOICES_EXPERIMENTAL = EMO_CHOICES_ALL if LOAD_QWEN_EMO else EMO_CHOICES_OFFICIAL

os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)

MAX_LENGTH_TO_USE_SPEED = 70
example_cases = []
EXAMPLE_VOICEPACK_PROFILES = {
    "voice_01": {"voiceId": "young-female-clear-lively", "displayName": "青年-女声-清亮灵动", "gender": "female"},
    "voice_02": {"voiceId": "young-female-firm-emotive", "displayName": "青年-女声-坚定感性", "gender": "female"},
    "voice_03": {"voiceId": "young-female-bright-cheerful", "displayName": "青年-女声-明亮活泼", "gender": "female"},
    "voice_04": {"voiceId": "young-female-calm-intellectual", "displayName": "青年-女声-沉稳知性", "gender": "female"},
    "voice_05": {"voiceId": "middle-aged-male-deep-narrative", "displayName": "中年-男声-厚重叙事", "gender": "male"},
    "voice_06": {"voiceId": "middle-aged-male-magnetic-witty", "displayName": "中年-男声-磁性诙谐", "gender": "male"},
    "voice_07": {"voiceId": "middle-aged-male-serious-steady", "displayName": "中年-男声-严肃沉稳", "gender": "male"},
    "voice_08": {"voiceId": "young-female-soft-emotive", "displayName": "青年-女声-柔和感性", "gender": "female"},
    "voice_09": {"voiceId": "young-female-sweet-playful", "displayName": "青年-女声-甜美俏皮", "gender": "female"},
    "voice_11": {"voiceId": "young-female-gentle-melancholic", "displayName": "青年-女声-温婉忧郁", "gender": "female"},
    "voice_12": {"voiceId": "middle-aged-male-deep-stern", "displayName": "中年-男声-低沉冷峻", "gender": "male"},
}
with open("examples/cases.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        example = json.loads(line)
        if example.get("emo_audio",None):
            emo_audio_path = os.path.join("examples",example["emo_audio"])
        else:
            emo_audio_path = None

        case = [os.path.join("examples", example.get("prompt_audio", "sample_prompt.wav")),
                              EMO_CHOICES_ALL[example.get("emo_mode",0)],
                              example.get("text"),
                             emo_audio_path,
                             example.get("emo_weight",1.0),
                             example.get("emo_text",""),
                             example.get("emo_vec_1",0),
                             example.get("emo_vec_2",0),
                             example.get("emo_vec_3",0),
                             example.get("emo_vec_4",0),
                             example.get("emo_vec_5",0),
                             example.get("emo_vec_6",0),
                             example.get("emo_vec_7",0),
                             example.get("emo_vec_8",0),
                             ]
        # v2.5 needs the language per-example so ja/es/ar demos auto-select it
        if IS_V25:
            case.append(example.get("lang", "ZH"))
        example_cases.append(case)

def get_example_cases(include_experimental = False):
    if include_experimental and LOAD_QWEN_EMO:
        cases = example_cases  # show every example
    else:
        # exclude emotion control mode 3 (emotion from text description)
        cases = [x for x in example_cases if x[1] != EMO_CHOICES_ALL[3]]

    if IS_V25:
        # The first UI column represents the installed voice pack, not its source WAV.
        return [
            [_example_voicepack_label(case[0]), *case[1:]]
            for case in cases
        ]
    return cases

def format_glossary_markdown():
    """将词汇表转换为Markdown表格格式"""
    if IS_V25 or not hasattr(tts, 'normalizer') or not tts.normalizer.term_glossary:
        return i18n("暂无术语")

    lines = [f"| {i18n('术语')} | {i18n('中文读法')} | {i18n('英文读法')} |"]
    lines.append("|---|---|---|")

    for term, reading in tts.normalizer.term_glossary.items():
        zh = reading.get("zh", "") if isinstance(reading, dict) else reading
        en = reading.get("en", "") if isinstance(reading, dict) else reading
        lines.append(f"| {term} | {zh} | {en} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Preset management
# ---------------------------------------------------------------------------

def _build_preset_data(
    emo_control_method,
    emo_weight,
    vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
    emo_text,
    emo_random,
    do_sample,
    top_p,
    top_k,
    temperature,
    length_penalty,
    num_beams,
    repetition_penalty,
    max_mel_tokens,
    max_text_tokens_per_segment,
):
    """Collect all non-audio UI values into a dictionary for preset storage."""
    return {
        "emo_control_method": int(emo_control_method) if emo_control_method is not None else 0,
        "emo_weight": float(emo_weight) if emo_weight is not None else 0.65,
        "emo_vector": [
            float(v) if v is not None else 0.0
            for v in [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        ],
        "emo_text": emo_text or "",
        "emo_random": bool(emo_random),
        "advanced_params": {
            "do_sample": bool(do_sample),
            "top_p": float(top_p) if top_p is not None else 0.8,
            "top_k": int(top_k) if top_k is not None else 30,
            "temperature": float(temperature) if temperature is not None else 0.8,
            "length_penalty": float(length_penalty) if length_penalty is not None else 0.0,
            "num_beams": int(num_beams) if num_beams is not None else 3,
            "repetition_penalty": float(repetition_penalty) if repetition_penalty is not None else 10.0,
            "max_mel_tokens": int(max_mel_tokens) if max_mel_tokens is not None else 1500,
            "max_text_tokens_per_segment": int(max_text_tokens_per_segment)
            if max_text_tokens_per_segment is not None else 120,
        },
    }


def on_preset_save(
    name,
    prompt_audio,
    emo_control_method,
    emo_ref_path,
    emo_weight,
    vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
    emo_text,
    emo_random,
    do_sample,
    top_p,
    top_k,
    temperature,
    length_penalty,
    num_beams,
    repetition_penalty,
    max_mel_tokens,
    max_text_tokens_per_segment,
):
    """Save the current UI state as a named preset."""
    name = name.strip() if name else ""
    if not name:
        gr.Warning(i18n("预设名称不能为空"))
        return gr.update()

    data = _build_preset_data(
        emo_control_method, emo_weight,
        vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
        emo_text, emo_random,
        do_sample, top_p, top_k, temperature,
        length_penalty, num_beams, repetition_penalty,
        max_mel_tokens, max_text_tokens_per_segment,
    )

    existed = name in list_presets()
    try:
        save_preset(name, data, prompt_audio=prompt_audio, emo_audio=emo_ref_path)
        msg = i18n("预设名称已存在，已覆盖") if existed else i18n("预设已保存")
        gr.Info(msg, duration=2)
    except Exception as e:
        gr.Error(f"{i18n('加载预设失败')}: {e}")
        return gr.update(), gr.update()

    choices = [""] + list_presets()
    has_presets = len(choices) > 1
    return (
        gr.update(choices=choices, value=name, interactive=has_presets),
        gr.update(choices=choices, value=name, interactive=has_presets),
    )


def on_preset_load(name):
    """Load a preset and return updates for all relevant UI components."""
    if not name:
        return {}

    data = load_preset(name)
    if data is None:
        gr.Warning(i18n("预设不存在"))
        return {}

    try:
        emo_method = int(data.get("emo_control_method", 0))
        is_experimental = emo_method == 3

        # Resolve audio paths; warn if files are missing.
        prompt_path = data.get("prompt_audio", "") or None
        emo_path = data.get("emo_audio", "") or None
        missing = []
        if prompt_path and not os.path.exists(prompt_path):
            missing.append("prompt")
            prompt_path = None
        if emo_path and not os.path.exists(emo_path):
            missing.append("emotion")
            emo_path = None
        if missing:
            gr.Warning(i18n("参考音频文件缺失，已跳过"))

        emo_weight_value = data.get("emo_weight", 0.65)
        emo_vec = data.get("emo_vector", [0.0] * 8)
        emo_text_value = data.get("emo_text", "")
        emo_random_value = data.get("emo_random", False)
        advanced = data.get("advanced_params", {})

        # Ensure the emotion vector has 8 elements.
        if len(emo_vec) < 8:
            emo_vec = emo_vec + [0.0] * (8 - len(emo_vec))
        elif len(emo_vec) > 8:
            emo_vec = emo_vec[:8]

        # Update the radio choices when loading an experimental preset.
        emo_choices = EMO_CHOICES_EXPERIMENTAL if is_experimental else EMO_CHOICES_OFFICIAL
        if emo_method >= len(emo_choices):
            gr.Warning(i18n("该预设使用了情感描述文本控制，当前显存不足未加载该模型，已重置为默认方式"))
            emo_method = 0
            is_experimental = False
            emo_choices = EMO_CHOICES_OFFICIAL

        return {
            experimental_checkbox: gr.update(value=is_experimental),
            emo_control_method: gr.update(choices=emo_choices, value=emo_choices[emo_method]),
            prompt_audio: gr.update(value=prompt_path),
            emo_upload: gr.update(value=emo_path),
            emo_weight: gr.update(value=emo_weight_value),
            vec1: gr.update(value=emo_vec[0]),
            vec2: gr.update(value=emo_vec[1]),
            vec3: gr.update(value=emo_vec[2]),
            vec4: gr.update(value=emo_vec[3]),
            vec5: gr.update(value=emo_vec[4]),
            vec6: gr.update(value=emo_vec[5]),
            vec7: gr.update(value=emo_vec[6]),
            vec8: gr.update(value=emo_vec[7]),
            emo_text: gr.update(value=emo_text_value),
            emo_random: gr.update(value=emo_random_value),
            do_sample: gr.update(value=advanced.get("do_sample", True)),
            top_p: gr.update(value=advanced.get("top_p", 0.8)),
            top_k: gr.update(value=advanced.get("top_k", 30)),
            temperature: gr.update(value=advanced.get("temperature", 0.8)),
            length_penalty: gr.update(value=advanced.get("length_penalty", 0.0)),
            num_beams: gr.update(value=advanced.get("num_beams", 3)),
            repetition_penalty: gr.update(value=advanced.get("repetition_penalty", 10.0)),
            max_mel_tokens: gr.update(value=advanced.get("max_mel_tokens", 1500)),
            max_text_tokens_per_segment: gr.update(
                value=advanced.get("max_text_tokens_per_segment", 120)
            ),
        }
    except Exception as e:
        gr.Error(f"{i18n('加载预设失败')}: {e}")
        return {}


def on_preset_delete(name):
    """Delete the selected preset and refresh the dropdown."""
    if not name:
        gr.Warning(i18n("请选择预设"))
        return gr.update()

    if delete_preset(name):
        gr.Info(i18n("预设已删除"), duration=2)
    else:
        gr.Warning(i18n("预设不存在"))

    choices = [""] + list_presets()
    has_presets = len(choices) > 1
    return (
        gr.update(value="", choices=choices, interactive=has_presets),
        gr.update(value="", choices=choices, interactive=has_presets),
    )


def update_save_preset_button(prompt_audio):
    """Enable the save button only when a voice reference audio is uploaded."""
    return gr.update(interactive=bool(prompt_audio))


def update_export_voicepack_button(prompt_audio):
    return gr.update(interactive=IS_V25 and bool(prompt_audio))


def _load_current_voicepack(pack_path):
    """Load a managed pack once, including source-model compatibility checks."""
    if not IS_V25 or voicepack_builder is None:
        raise ValueError("Voice packs require IndexTTS-2.5")
    resolved = os.path.abspath(pack_path)
    managed_roots = (VOICEPACK_DIR, VOICEPACK_EXPORT_DIR)
    if all(os.path.commonpath([root, resolved]) != root for root in managed_roots):
        raise ValueError(i18n("只能选择应用音色包目录内的文件"))
    stat = os.stat(resolved)
    cache_key = (resolved, stat.st_mtime_ns, stat.st_size)
    cached = _voicepack_cache.get(cache_key)
    if cached is not None:
        return cached
    pack = load_voicepack(
        resolved,
        expected_model_fingerprint=voicepack_builder.source_model_fingerprint(),
    )
    _voicepack_cache.clear()
    _voicepack_cache[cache_key] = pack
    return pack


def _voicepack_label(pack):
    gender_labels = {
        "female": i18n("女声"),
        "male": i18n("男声"),
        "neutral": i18n("中性"),
        "unknown": i18n("未知"),
    }
    manifest = pack.manifest
    return f"{manifest['displayName']} · {manifest['voiceId']} · {gender_labels[manifest['gender']]}"


def list_application_voicepacks():
    """Return valid packs installed in the WebUI-managed application directory."""
    packs = []
    for filename in sorted(os.listdir(VOICEPACK_DIR)):
        if not filename.lower().endswith(".ivp"):
            continue
        path = os.path.join(VOICEPACK_DIR, filename)
        try:
            pack = _load_current_voicepack(path)
        except Exception as exc:
            print(f"Skipping invalid voice pack {path}: {exc}")
            continue
        packs.append((pack, path))
    return packs


def voicepack_choices():
    return [(_voicepack_label(pack), path) for pack, path in list_application_voicepacks()]


def format_voicepack_selection(pack_path):
    if not pack_path:
        return i18n("未选择音色包")
    try:
        pack = _load_current_voicepack(pack_path)
    except Exception as exc:
        return f"❌ {i18n('音色包不可用')}：{exc}"
    manifest = pack.manifest
    return manifest["displayName"]


def refresh_voicepack_selection(selected=None, *, select_first=False):
    if selected and os.path.isfile(selected):
        try:
            _load_current_voicepack(selected)
            return selected, format_voicepack_selection(selected)
        except Exception:
            selected = ""

    choices = voicepack_choices()
    values = [value for _, value in choices]
    selected = values[0] if select_first and values else ""
    return selected, format_voicepack_selection(selected)


def import_voicepack_from_webui(uploaded_pack):
    """Validate, install and select a dropped ``.ivp`` file."""
    if not uploaded_pack:
        return (*refresh_voicepack_selection(), gr.update())
    try:
        pack = load_voicepack(
            uploaded_pack,
            expected_model_fingerprint=voicepack_builder.source_model_fingerprint(),
        )
        destination = os.path.join(VOICEPACK_DIR, f"{pack.voice_id}.ivp")
        source = os.path.abspath(uploaded_pack)
        if source != destination:
            temporary = destination + ".importing"
            try:
                shutil.copyfile(source, temporary)
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.remove(temporary)
        _voicepack_cache.clear()
        selected, details = refresh_voicepack_selection(destination)
        gr.Info(i18n("音色包已安装并选中"), duration=2)
        return selected, details, gr.update(value=None)
    except Exception as exc:
        print(f"Failed to import voice pack: {exc}")
        raise gr.Error(f"{i18n('导入音色包失败')}: {exc}") from exc


def _example_voicepack_path(audio_path):
    voice_id = _example_voicepack_metadata(audio_path)["voiceId"]
    return os.path.join(VOICEPACK_DIR, f"{voice_id}.ivp")


def _example_voicepack_metadata(audio_path):
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    profile = EXAMPLE_VOICEPACK_PROFILES.get(stem)
    if profile is None:
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._") or "voice"
        return {
            "voiceId": f"builtin-{safe_stem}"[:64],
            "displayName": f"成年-未知-{stem}",
            "gender": "unknown",
        }
    return {
        "voiceId": profile["voiceId"],
        "displayName": profile["displayName"],
        "gender": profile["gender"],
    }


def _example_voicepack_label(audio_path):
    metadata = _example_voicepack_metadata(audio_path)
    return f"{metadata['displayName']} · {metadata['voiceId']}"


def _example_voicepack_path_from_label(label):
    for case in example_cases:
        if _example_voicepack_label(case[0]) == label:
            return _example_voicepack_path(case[0])
    raise gr.Error(i18n("音色包不可用"))


def _build_example_voicepack(audio_path):
    output_path = _example_voicepack_path(audio_path)
    metadata = _example_voicepack_metadata(audio_path)
    if os.path.isfile(output_path):
        try:
            existing = _load_current_voicepack(output_path)
            if (
                existing.manifest.get("voiceId") == metadata["voiceId"]
                and existing.manifest.get("displayName") == metadata["displayName"]
                and existing.manifest.get("gender") == metadata["gender"]
            ):
                return output_path
        except Exception:
            pass
    with mutex:
        voicepack_builder.build(
            audio_path,
            metadata,
            output_path,
        )
    _voicepack_cache.clear()
    return output_path


def prebuild_example_voicepacks():
    """Prepare every official example voice before the UI becomes interactive."""
    if not IS_V25:
        return
    unique_audio = list(dict.fromkeys(example[0] for example in example_cases))
    print(f">> Preparing {len(unique_audio)} example voice packs...")
    for index, audio_path in enumerate(unique_audio, start=1):
        output_path = _build_example_voicepack(audio_path)
        print(f">> Example voice pack {index}/{len(unique_audio)} ready: {output_path}")
    for audio_path in unique_audio:
        stem = os.path.splitext(os.path.basename(audio_path))[0]
        legacy_path = os.path.join(VOICEPACK_DIR, f"example-{stem}.ivp")
        if os.path.isfile(legacy_path) and os.path.abspath(legacy_path) != os.path.abspath(_example_voicepack_path(audio_path)):
            os.remove(legacy_path)
            print(f">> Removed legacy example voice pack: {legacy_path}")
    _voicepack_cache.clear()


def export_voicepack_from_webui(prompt_audio, voice_id, display_name, gender):
    """Create a reusable conditioning pack without storing the reference WAV."""
    if not IS_V25 or voicepack_builder is None:
        raise gr.Error(i18n("导出音色包仅支持 IndexTTS-2.5"))
    if not prompt_audio:
        raise gr.Error(i18n("请先上传音色参考音频"))
    voice_id = (voice_id or "").strip()
    display_name = (display_name or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", voice_id):
        raise gr.Error(i18n("音色 ID 格式无效"))
    if not display_name:
        raise gr.Error(i18n("音色名称不能为空"))
    output_path = os.path.join(VOICEPACK_EXPORT_DIR, f"{voice_id}.ivp")
    try:
        with mutex:
            path = voicepack_builder.build(
                prompt_audio,
                {"voiceId": voice_id, "displayName": display_name, "gender": gender},
                output_path,
            )
    except Exception as exc:
        print(f"Failed to export voice pack: {exc}")
        raise gr.Error(f"{i18n('导出音色包失败')}: {exc}") from exc
    _voicepack_cache.clear()
    selection, details = refresh_voicepack_selection(str(path))
    return str(path), f"✅ {i18n('音色包已生成')}：`{path}`", selection, details


def update_delete_preset_button(preset_name):
    """Enable the delete button only when a preset is selected."""
    return gr.update(interactive=bool(preset_name))


def format_preset_details(name):
    """Render a preset's parameters as a Markdown table for the management tab."""
    if not name:
        return i18n("请选择要管理的预设")

    data = load_preset(name)
    if data is None:
        return i18n("预设不存在")

    try:
        emo_method = int(data.get("emo_control_method", 0))
        emo_label = EMO_CHOICES_ALL[emo_method] if 0 <= emo_method < len(EMO_CHOICES_ALL) else i18n("未知")
        emo_weight = data.get("emo_weight", 0.0)
        emo_vec = data.get("emo_vector", [0.0] * 8)
        emo_text = data.get("emo_text", "") or i18n("无")
        emo_random = data.get("emo_random", False)
        advanced = data.get("advanced_params", {})
        prompt_path = data.get("prompt_audio", "") or i18n("无")
        emo_path = data.get("emo_audio", "") or i18n("无")

        lines = [
            f"### {i18n('预设详情')}: {name}",
            "",
            f"| {i18n('属性')} | {i18n('值')} |",
            "|---|---|",
            f"| {i18n('名称')} | {name} |",
            f"| {i18n('情感控制方式')} | {emo_label} |",
            f"| {i18n('情感权重')} | {emo_weight} |",
            f"| {i18n('情感随机采样')} | {'On' if emo_random else 'Off'} |",
            f"| {i18n('音色音频')} | `{prompt_path}` |",
            f"| {i18n('情感音频')} | `{emo_path}` |",
            "",
            f"**{i18n('情感向量')}**: `[{', '.join(str(round(v, 2)) for v in emo_vec)}]`",
            "",
            f"**{i18n('情感描述文本')}**: {emo_text}",
            "",
            f"**{i18n('高级生成参数设置')}**:",
            "",
        ]
        for key, value in advanced.items():
            lines.append(f"- `{key}`: {value}")
        return "\n".join(lines)
    except Exception as e:
        return f"{i18n('加载预设失败')}: {e}"


def refresh_preset_choices():
    """Return fresh choices and interactive state for all preset dropdowns."""
    choices = [""] + list_presets()
    has_presets = len(choices) > 1
    return (
        gr.update(choices=choices, value="", interactive=has_presets),
        gr.update(choices=choices, value="", interactive=has_presets),
    )


def _format_preset_preview(
    prompt_audio,
    emo_control_method,
    emo_ref_path,
    emo_weight,
    vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
    emo_text,
    emo_random,
    do_sample,
    top_p,
    top_k,
    temperature,
    length_penalty,
    num_beams,
    repetition_penalty,
    max_mel_tokens,
    max_text_tokens_per_segment,
):
    """Format the current UI state as a Markdown preview for the save modal."""
    emo_label = EMO_CHOICES_ALL[emo_control_method] if 0 <= emo_control_method < len(EMO_CHOICES_ALL) else i18n("未知")
    vec = [float(v) for v in [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]]
    prompt_label = os.path.basename(prompt_audio) if prompt_audio else i18n("无")
    emo_label_audio = os.path.basename(emo_ref_path) if emo_ref_path else i18n("无")

    lines = [
        f"**{i18n('情感控制方式')}**: {emo_label}",
        f"**{i18n('情感权重')}**: {emo_weight}",
        f"**{i18n('情感随机采样')}**: {'On' if emo_random else 'Off'}",
        f"**{i18n('音色音频')}**: `{prompt_label}`",
        f"**{i18n('情感音频')}**: `{emo_label_audio}`",
        "",
        f"**{i18n('情感向量')}**: `[{', '.join(str(round(v, 2)) for v in vec)}]`",
        f"**{i18n('情感描述文本')}**: {emo_text or i18n('无')}",
        "",
        f"**{i18n('高级生成参数设置')}**:",
        f"- do_sample: {do_sample}",
        f"- top_p: {top_p}",
        f"- top_k: {top_k}",
        f"- temperature: {temperature}",
        f"- length_penalty: {length_penalty}",
        f"- num_beams: {num_beams}",
        f"- repetition_penalty: {repetition_penalty}",
        f"- max_mel_tokens: {max_mel_tokens}",
        f"- max_text_tokens_per_segment: {max_text_tokens_per_segment}",
    ]
    return "\n".join(lines)


def open_save_preset_modal(
    prompt_audio,
    emo_control_method,
    emo_ref_path,
    emo_weight,
    vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
    emo_text,
    emo_random,
    do_sample,
    top_p,
    top_k,
    temperature,
    length_penalty,
    num_beams,
    repetition_penalty,
    max_mel_tokens,
    max_text_tokens_per_segment,
):
    """Open the save-preset modal and populate the preview."""
    preview = _format_preset_preview(
        prompt_audio, emo_control_method, emo_ref_path, emo_weight,
        vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
        emo_text, emo_random,
        do_sample, top_p, top_k, temperature,
        length_penalty, num_beams, repetition_penalty,
        max_mel_tokens, max_text_tokens_per_segment,
    )
    return (
        gr.update(visible=True),
        gr.update(value=preview),
        gr.update(value=""),
    )


def confirm_save_preset_from_modal(
    name,
    prompt_audio,
    emo_control_method,
    emo_ref_path,
    emo_weight,
    vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
    emo_text,
    emo_random,
    do_sample,
    top_p,
    top_k,
    temperature,
    length_penalty,
    num_beams,
    repetition_penalty,
    max_mel_tokens,
    max_text_tokens_per_segment,
):
    """Save the preset and close the modal."""
    result = on_preset_save(
        name,
        prompt_audio,
        emo_control_method,
        emo_ref_path,
        emo_weight,
        vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
        emo_text,
        emo_random,
        do_sample,
        top_p,
        top_k,
        temperature,
        length_penalty,
        num_beams,
        repetition_penalty,
        max_mel_tokens,
        max_text_tokens_per_segment,
    )
    return (
        gr.update(visible=False),  # modal
        result[0],  # load_preset_dropdown
        result[1],  # manage_preset_dropdown
    )


def close_save_preset_modal():
    """Close the save-preset modal without saving."""
    return gr.update(visible=False)


def gen_single(emo_control_method, prompt, selected_voicepack, text,
               lang_choice,
               emo_ref_path, emo_weight,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_text,emo_random,
               max_text_tokens_per_segment=120,
               duration_factor=1.0,
                *args, progress=gr.Progress()):
    output_path = None
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    # set gradio progress
    tts.gr_progress = progress
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args

    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
        # "typical_sampling": bool(typical_sampling),
        # "typical_mass": float(typical_mass),
    }
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:  # emotion from speaker
        emo_ref_path = None  # remove external reference audio
    if emo_control_method == 1:  # emotion from reference audio
        pass
    if emo_control_method == 2:  # emotion from custom vectors
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec = tts.normalize_emo_vec(vec, apply_bias=True)
    else:
        # don't use the emotion vector inputs for the other modes
        vec = None

    if emo_text == "":
        # erase empty emotion descriptions; `infer()` will then automatically use the main prompt
        emo_text = None

    print(f"Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}")
    voice_conditioning = None
    if selected_voicepack:
        try:
            voice_conditioning = _load_current_voicepack(selected_voicepack).tensors
        except Exception as exc:
            raise gr.Error(f"{i18n('音色包不可用')}：{exc}") from exc
    elif IS_V25:
        raise gr.Error(i18n("请选择音色包"))
    elif not prompt:
        raise gr.Error(i18n("请选择音色包或上传音色参考音频"))

    infer_kwargs = dict(
        spk_audio_prompt=None if voice_conditioning is not None else prompt, text=text,
        output_path=output_path,
        emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
        emo_vector=vec,
        use_emo_text=(emo_control_method==3), emo_text=emo_text, use_random=emo_random,
        verbose=cmd_args.verbose,
        max_text_tokens_per_segment=int(max_text_tokens_per_segment),
        duration_factor=float(duration_factor),
        **kwargs,
    )
    if IS_V25:
        infer_kwargs["lang"] = lang_choice or "ZH"
        infer_kwargs["voice_conditioning"] = voice_conditioning
    output = tts.infer(**infer_kwargs)
    return gr.update(value=output,visible=True)

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

def create_warning_message(warning_text):
    return gr.HTML(f"<div style=\"padding: 0.5em 0.8em; border-radius: 0.5em; background: #ffa87d; color: #000; font-weight: bold\">{html.escape(warning_text)}</div>")

def create_experimental_warning_message():
    return create_warning_message(i18n('提示：此功能为实验版，结果尚不稳定，我们正在持续优化中。'))

prebuild_example_voicepacks()

with gr.Blocks(
    title=f"IndexTTS-{cmd_args.version} Demo",
    css="""
        /* Make the voice reference audio upload area more compact. */
        #prompt_audio_compact .audio-container,
        #prompt_audio_compact .upload-container {
            min-height: 110px !important;
        }
        #prompt_audio_compact .empty {
            min-height: 80px !important;
        }

        /* Modal overlay for saving presets. */
        .preset-modal-overlay {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            background: rgba(0, 0, 0, 0.5) !important;
            z-index: 1000 !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        .preset-modal-overlay > .column-wrap,
        .preset-modal-overlay > div {
            width: auto !important;
            height: auto !important;
        }
        .preset-modal-content {
            background: var(--body-background-fill) !important;
            padding: 24px !important;
            border-radius: 12px !important;
            width: 90vw !important;
            max-width: 560px !important;
            max-height: 80vh !important;
            overflow-y: auto !important;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2) !important;
        }
        .preset-modal-content > .column-wrap,
        .preset-modal-content > div {
            gap: 16px !important;
        }
    """,
) as demo:
    arxiv_id = "2601.03888" if IS_V25 else "2506.21619"
    gr.HTML(f'''
       <h2 style="text-align:center">IndexTTS-{cmd_args.version}</h2>
   <p align="center">
   <a href='https://arxiv.org/abs/{arxiv_id}'><img src='https://img.shields.io/badge/ArXiv-{arxiv_id}-red'></a>
   </p>
       ''')

    with gr.Tab(i18n("音频生成")):
        os.makedirs("prompts", exist_ok=True)

        if IS_V25:
            _initial_voicepack_choices = voicepack_choices()
            _initial_voicepack = (
                _initial_voicepack_choices[0][1]
                if _initial_voicepack_choices
                else ""
            )
            gr.Markdown(f"### {i18n('当前应用音色包')}")
            selected_voicepack = gr.State(value=_initial_voicepack)
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    selected_voicepack_status = gr.Markdown(
                        format_voicepack_selection(_initial_voicepack),
                        container=True,
                        min_height=110,
                    )
                voicepack_import = gr.File(
                    label=i18n("拖入 .ivp 音色包以安装并选中"),
                    show_label=False,
                    file_types=[".ivp"],
                    type="filepath",
                    height=110,
                    scale=1,
                    elem_id="voicepack_drop",
                )
        else:
            gr.Markdown(f"### {i18n('音色参考音频')}")
            with gr.Row(equal_height=False):
                prompt_audio = gr.Audio(
                    label="",
                    key="prompt_audio",
                    sources=["upload", "microphone"],
                    type="filepath",
                    elem_classes=["compact-audio"],
                    elem_id="prompt_audio_compact",
                )
            selected_voicepack = gr.State(value="")
            save_preset_btn = gr.Button(i18n("保存为预设"), interactive=False)
            _has_presets = bool(list_presets())
            load_preset_dropdown = gr.Dropdown(
                choices=[""] + list_presets(),
                value="",
                label=i18n("从预设加载"),
                info=i18n("从预设加载音色和参数"),
                allow_custom_value=False,
                interactive=_has_presets,
            )

        # Text input and generation section
        gr.Markdown(f"### {i18n('文本')}")
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                input_text_single = gr.TextArea(
                    label="",
                    key="input_text_single",
                    placeholder=i18n("请输入目标文本"),
                    info=f"{i18n('当前模型版本')}{tts.model_version or '1.0'}",
                    lines=5,
                )
                if IS_V25:
                    lang_dropdown = gr.Dropdown(
                        choices=["ZH", "EN", "JA", "AR", "ES"],
                        value="ZH",
                        label=i18n("语言"),
                    )
                else:
                    lang_dropdown = gr.State(value=None)
                duration_factor = gr.Slider(
                    label=i18n("时长系数"), minimum=0.5, maximum=2.0, value=1.0, step=0.01,
                    info=f'{i18n("快")} ← — {i18n("不变")} — → {i18n("慢")}',
                    key="duration_factor",
                )
            with gr.Column(scale=1):
                gen_button = gr.Button(
                    i18n("生成语音"), key="gen_button", interactive=True
                )
                output_audio = gr.Audio(
                    label=i18n("生成结果"), visible=True, key="output_audio"
                )

        with gr.Row():
            experimental_checkbox = gr.Checkbox(label=i18n("显示实验功能"), value=False)
            _has_glossary = not IS_V25 and hasattr(tts, 'normalizer')
            glossary_checkbox = gr.Checkbox(label=i18n("开启术语词汇读音"), value=tts.normalizer.enable_glossary if _has_glossary else False, visible=_has_glossary)
        with gr.Accordion(i18n("功能设置")):
            # 情感控制选项部分
            with gr.Row():
                emo_control_method = gr.Radio(
                    choices=EMO_CHOICES_OFFICIAL,
                    type="index",
                    value=EMO_CHOICES_OFFICIAL[0],label=i18n("情感控制方式"))
                # we MUST have an extra, INVISIBLE list of *all* emotion control
                # methods so that gr.Dataset() can fetch ALL control mode labels!
                # otherwise, the gr.Dataset()'s experimental labels would be empty!
                emo_control_method_all = gr.Radio(
                    choices=EMO_CHOICES_ALL,
                    type="index",
                    value=EMO_CHOICES_ALL[0], label=i18n("情感控制方式"),
                    visible=False)  # do not render
        # 情感参考音频部分
        with gr.Group(visible=False) as emotion_reference_group:
            with gr.Row():
                emo_upload = gr.Audio(label=i18n("上传情感参考音频"), type="filepath")

        # 情感随机采样
        with gr.Row(visible=False) as emotion_randomize_group:
            emo_random = gr.Checkbox(label=i18n("情感随机采样"), value=False)

        # 情感向量控制部分
        with gr.Group(visible=False) as emotion_vector_group:
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label=i18n("喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec2 = gr.Slider(label=i18n("怒"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec3 = gr.Slider(label=i18n("哀"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec4 = gr.Slider(label=i18n("惧"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                with gr.Column():
                    vec5 = gr.Slider(label=i18n("厌恶"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec6 = gr.Slider(label=i18n("低落"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec7 = gr.Slider(label=i18n("惊喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec8 = gr.Slider(label=i18n("平静"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)

        with gr.Group(visible=False) as emo_text_group:
            create_experimental_warning_message()
            with gr.Row():
                emo_text = gr.Textbox(label=i18n("情感描述文本"),
                                      placeholder=i18n("请输入情绪描述（或留空以自动使用目标文本作为情绪描述）"),
                                      value="",
                                      info=i18n("例如：委屈巴巴、危险在悄悄逼近"))

        with gr.Row(visible=False) as emo_weight_group:
            emo_weight = gr.Slider(label=i18n("情感权重"), minimum=0.0, maximum=1.0, value=0.65, step=0.01)

        # 术语词汇表管理
        with gr.Accordion(i18n("自定义术语词汇读音"), open=False, visible=_has_glossary and tts.normalizer.enable_glossary) as glossary_accordion:
            gr.Markdown(i18n("自定义个别专业术语的读音"))
            with gr.Row():
                with gr.Column(scale=1):
                    glossary_term = gr.Textbox(
                        label=i18n("术语"),
                        placeholder="IndexTTS2",
                    )
                    glossary_reading_zh = gr.Textbox(
                        label=i18n("中文读法"),
                        placeholder="Index T-T-S 二",
                    )
                    glossary_reading_en = gr.Textbox(
                        label=i18n("英文读法"),
                        placeholder="Index T-T-S two",
                    )
                    btn_add_term = gr.Button(i18n("添加术语"), scale=1)
                with gr.Column(scale=2):
                    glossary_table = gr.Markdown(
                        value=format_glossary_markdown()
                    )

        with gr.Accordion(i18n("高级生成参数设置"), open=False, visible=True) as advanced_settings_group:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"**{i18n('GPT2 采样设置')}** _{i18n('参数会影响音频多样性和生成速度详见')} [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True, info=i18n("是否进行采样"))
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                        length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                    max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info=i18n("生成Token最大数量，过小导致音频被截断"), key="max_mel_tokens")
                    # with gr.Row():
                    #     typical_sampling = gr.Checkbox(label="typical_sampling", value=False, info="不建议使用")
                    #     typical_mass = gr.Slider(label="typical_mass", value=0.9, minimum=0.0, maximum=1.0, step=0.1)
                with gr.Column(scale=2):
                    gr.Markdown(f'**{i18n("分句设置")}** _{i18n("参数会影响音频质量和生成速度")}_')
                    with gr.Row():
                        initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                        max_text_tokens_per_segment = gr.Slider(
                            label=i18n("分句最大Token数"), value=initial_value, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, key="max_text_tokens_per_segment",
                            info=i18n("建议80~200之间，值越大，分句越长；值越小，分句越碎；过小过大都可能导致音频质量不高"),
                        )
                    with gr.Accordion(i18n("预览分句结果"), open=True) as segments_settings:
                        segments_preview = gr.Dataframe(
                            headers=[i18n("序号"), i18n("分句内容"), i18n("Token数")],
                            key="segments_preview",
                            wrap=True,
                        )
            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                # typical_sampling, typical_mass,
            ]

        # we must use `gr.Dataset` to support dynamic UI rewrites, since `gr.Examples`
        # binds tightly to UI and always restores the initial state of all components,
        # such as the list of available choices in emo_control_method.
        # these components are NOT "connected". it just reads the column labels/available
        # states from them, so we MUST link to the "all options" versions of all components,
        # such as `emo_control_method_all` (to be able to see EXPERIMENTAL text labels)!
        example_voice_component = gr.Textbox(label=i18n("音色包"), render=False)
        example_components = [example_voice_component,
                        emo_control_method_all,  # important: support all mode labels!
                        input_text_single,
                        emo_upload,
                        emo_weight,
                        emo_text,
                        vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        # v2.5: append the language column so ja/es/ar examples auto-select it
        if IS_V25:
            example_components.append(lang_dropdown)
        example_table = gr.Dataset(label=i18n("快速设置（自动选择对应音色包）"),
            samples_per_page=20,
            samples=get_example_cases(include_experimental=False),
            type="values",
            components=example_components
        )

    if IS_V25:
        with gr.Tab(i18n("音色包制作")):
            gr.Markdown(f"## {i18n('从参考音频制作音色包')}")
            gr.Markdown(i18n("制作完成后会自动安装到应用音色包列表，并选中该音色。"))
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_audio = gr.Audio(
                        label=i18n("音色参考音频"),
                        key="voicepack_reference_audio",
                        sources=["upload", "microphone"],
                        type="filepath",
                        elem_classes=["compact-audio"],
                        elem_id="prompt_audio_compact",
                    )
                with gr.Column(scale=1):
                    voicepack_id = gr.Textbox(
                        label=i18n("音色 ID"),
                        value="reader-voice-01",
                        info=i18n("仅可使用字母、数字、点、下划线和连字符"),
                    )
                    voicepack_name = gr.Textbox(
                        label=i18n("音色名称"),
                        value=i18n("阅读音色"),
                    )
                    voicepack_gender = gr.Dropdown(
                        choices=[
                            (i18n("女声"), "female"),
                            (i18n("男声"), "male"),
                            (i18n("中性"), "neutral"),
                            (i18n("未知"), "unknown"),
                        ],
                        value="unknown",
                        label=i18n("性别"),
                    )
                    export_voicepack_btn = gr.Button(
                        i18n("生成并安装音色包"),
                        interactive=False,
                        variant="primary",
                    )
            gr.Markdown(i18n("音色包只保存预计算张量，不包含参考音频。"))
            voicepack_status = gr.Markdown("")
            voicepack_download = gr.File(label=i18n("下载音色包"), interactive=False)

    if not IS_V25:
        with gr.Tab(i18n("预设管理")):
            gr.Markdown(f"## {i18n('预设管理')}")

            with gr.Row():
                _has_presets = bool(list_presets())
                manage_preset_dropdown = gr.Dropdown(
                    choices=[""] + list_presets(),
                    value="",
                    label=i18n("预设列表"),
                    allow_custom_value=False,
                    scale=2,
                    interactive=_has_presets,
                )
                apply_preset_btn = gr.Button(i18n("应用"), scale=1)
                delete_preset_btn = gr.Button(i18n("删除"), scale=1, interactive=False)
                refresh_preset_btn = gr.Button(i18n("刷新"), scale=1)

            preset_details_markdown = gr.Markdown(
                value=i18n("请选择要管理的预设"),
            )

            with gr.Accordion(i18n("从当前状态创建"), open=False):
                with gr.Row():
                    create_preset_name = gr.Textbox(
                        label=i18n("预设名称"),
                        placeholder=i18n("请输入预设名称"),
                        value="",
                        scale=2,
                    )
                    create_preset_btn = gr.Button(i18n("创建"), scale=1)

        # -------------------------------------------------------------------
        # Save Preset Modal (IndexTTS-2 legacy UI only)
        # -------------------------------------------------------------------
        with gr.Column(
            visible=False,
            elem_classes=["preset-modal-overlay"],
        ) as save_preset_modal:
            with gr.Column(elem_classes=["preset-modal-content"]):
                gr.Markdown(f"### {i18n('保存预设')}")
                modal_preset_preview = gr.Markdown(
                    label=i18n("预设预览"),
                    value=i18n("预设预览"),
                )
                modal_preset_name = gr.Textbox(
                    label=i18n("预设名称"),
                    placeholder=i18n("请输入预设名称"),
                    value="",
                )
                with gr.Row():
                    modal_cancel_btn = gr.Button(i18n("取消"), scale=1)
                    modal_confirm_btn = gr.Button(
                        i18n("确认"), scale=1, variant="primary"
                    )

    def on_example_click(example):
        print(f"Example clicked: ({len(example)} values) = {example!r}")
        parameter_updates = [
            gr.update(value=example[1]),
            gr.update(value=example[2]),
            gr.update(value=example[3]),
            gr.update(value=example[4]),
            gr.update(value=example[5]),
            gr.update(value=example[6]),
            gr.update(value=example[7]),
            gr.update(value=example[8]),
            gr.update(value=example[9]),
            gr.update(value=example[10]),
            gr.update(value=example[11]),
            gr.update(value=example[12]),
            gr.update(value=example[13]),
        ]
        if IS_V25:
            pack_path = _example_voicepack_path_from_label(example[0])
            _load_current_voicepack(pack_path)
            selection, details = refresh_voicepack_selection(pack_path)
            return [
                selection,
                *parameter_updates,
                gr.update(value=example[14]),
                details,
            ]
        return [gr.update(value=example[0]), *parameter_updates]

    # click() event works on both desktop and mobile UI
    if IS_V25:
        example_outputs = [
            selected_voicepack,
            emo_control_method,
            input_text_single,
            emo_upload,
            emo_weight,
            emo_text,
            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
            lang_dropdown,
            selected_voicepack_status,
        ]
    else:
        example_outputs = [
            prompt_audio,
            emo_control_method,
            input_text_single,
            emo_upload,
            emo_weight,
            emo_text,
            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
        ]
    example_table.click(on_example_click,
                        inputs=[example_table],
                        outputs=example_outputs
    )

    def on_input_text_change(text, max_text_tokens_per_segment, lang_choice):
        if text and len(text) > 0:
            if IS_V25:
                lang_prefix = f'<|{(lang_choice or "ZH").lower()}|> '
                segments = tts.split_text_by_tokens(
                    text,
                    int(max_text_tokens_per_segment),
                    lang_prefix,
                )
                data = []
                for i, segment_str in enumerate(segments):
                    tokens_count = len(tts.tokenizer.encode(
                        lang_prefix + segment_str,
                        allowed_special='all',
                    ))
                    data.append([i, segment_str, tokens_count])
            else:
                text_tokens_list = tts.tokenizer.tokenize(text)
                segments = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
                data = []
                for i, s in enumerate(segments):
                    segment_str = ''.join(s)
                    tokens_count = len(s)
                    data.append([i, segment_str, tokens_count])
            return {
                segments_preview: gr.update(value=data, visible=True, type="array"),
            }
        else:
            df = pd.DataFrame([], columns=[i18n("序号"), i18n("分句内容"), i18n("Token数")])
            return {
                segments_preview: gr.update(value=df),
            }

    # 术语词汇表事件处理函数
    def on_add_glossary_term(term, reading_zh, reading_en):
        """添加术语到词汇表并自动保存"""
        if IS_V25 or not hasattr(tts, 'normalizer'):
            return gr.update()
        term = term.rstrip()
        reading_zh = reading_zh.rstrip()
        reading_en = reading_en.rstrip()

        if not term:
            gr.Warning(i18n("请输入术语"))
            return gr.update()
            
        if not reading_zh and not reading_en:
            gr.Warning(i18n("请至少输入一种读法"))
            return gr.update()
        

        # 构建读法数据
        if reading_zh and reading_en:
            reading = {"zh": reading_zh, "en": reading_en}
        elif reading_zh:
            reading = {"zh": reading_zh}
        elif reading_en:
            reading = {"en": reading_en}
        else:
            reading = reading_zh or reading_en

        # 添加到词汇表
        tts.normalizer.term_glossary[term] = reading

        # 自动保存到文件
        try:
            tts.normalizer.save_glossary_to_yaml(tts.glossary_path)
            gr.Info(i18n("词汇表已更新"), duration=1)
        except Exception as e:
            gr.Error(i18n("保存词汇表时出错"))
            print(f"Error details: {e}")
            return gr.update()

        # 更新Markdown表格
        return gr.update(value=format_glossary_markdown())
        

    def on_method_change(emo_control_method):
        if emo_control_method == 1:  # emotion reference audio
            return (gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 2:  # emotion vectors
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 3:  # emotion text description
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True)
                    )
        else:  # 0: same as speaker voice
            return (gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )

    emo_control_method.change(on_method_change,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group,
                 emotion_randomize_group,
                 emotion_vector_group,
                 emo_text_group,
                 emo_weight_group]
    )

    def on_experimental_change(is_experimental, current_mode_index):
        # 切换情感控制选项
        new_choices = EMO_CHOICES_EXPERIMENTAL if is_experimental else EMO_CHOICES_OFFICIAL
        # if their current mode selection doesn't exist in new choices, reset to 0.
        # we don't verify that OLD index means the same in NEW list, since we KNOW it does.
        new_index = current_mode_index if current_mode_index < len(new_choices) else 0

        return (
            gr.update(choices=new_choices, value=new_choices[new_index]),
            gr.update(samples=get_example_cases(include_experimental=is_experimental)),
        )

    experimental_checkbox.change(
        on_experimental_change,
        inputs=[experimental_checkbox, emo_control_method],
        outputs=[emo_control_method, example_table]
    )

    def on_glossary_checkbox_change(is_enabled):
        """控制术语词汇表的可见性"""
        if IS_V25 or not hasattr(tts, 'normalizer'):
            return gr.update(visible=False)
        tts.normalizer.enable_glossary = is_enabled
        return gr.update(visible=is_enabled)

    glossary_checkbox.change(
        on_glossary_checkbox_change,
        inputs=[glossary_checkbox],
        outputs=[glossary_accordion]
    )

    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment, lang_dropdown],
        outputs=[segments_preview]
    )

    max_text_tokens_per_segment.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment, lang_dropdown],
        outputs=[segments_preview]
    )

    if IS_V25:
        lang_dropdown.change(
            on_input_text_change,
            inputs=[input_text_single, max_text_tokens_per_segment, lang_dropdown],
            outputs=[segments_preview]
        )

    if IS_V25:
        prompt_audio.change(
            update_export_voicepack_button,
            inputs=[prompt_audio],
            outputs=[export_voicepack_btn],
        )
        export_voicepack_btn.click(
            export_voicepack_from_webui,
            inputs=[prompt_audio, voicepack_id, voicepack_name, voicepack_gender],
            outputs=[
                voicepack_download,
                voicepack_status,
                selected_voicepack,
                selected_voicepack_status,
            ],
        )
        voicepack_import.upload(
            import_voicepack_from_webui,
            inputs=[voicepack_import],
            outputs=[selected_voicepack, selected_voicepack_status, voicepack_import],
        )
    else:
        prompt_audio.upload(
            update_prompt_audio,
            inputs=[],
            outputs=[gen_button],
        )
        prompt_audio.change(
            update_save_preset_button,
            inputs=[prompt_audio],
            outputs=[save_preset_btn],
        )

    def on_demo_load():
        """Refresh the active model's UI-managed resources."""
        if IS_V25:
            return (gr.update(), *refresh_voicepack_selection(select_first=True))
        if not hasattr(tts, 'normalizer'):
            return (gr.update(), *refresh_preset_choices())
        try:
            tts.normalizer.load_glossary_from_yaml(tts.glossary_path)
        except Exception as e:
            gr.Error(i18n("加载词汇表时出错"))
            print(f"Failed to reload glossary on page load: {e}")
        return (gr.update(value=format_glossary_markdown()),
                *refresh_preset_choices())

    # 术语词汇表事件绑定
    btn_add_term.click(
        on_add_glossary_term,
        inputs=[glossary_term, glossary_reading_zh, glossary_reading_en],
        outputs=[glossary_table]
    )

    # 页面加载时重新加载glossary并刷新预设列表
    if IS_V25:
        _demo_load_outputs = [
            glossary_table,
            selected_voicepack,
            selected_voicepack_status,
        ]
    else:
        _demo_load_outputs = [
            glossary_table,
            load_preset_dropdown,
            manage_preset_dropdown,
        ]
    demo.load(
        on_demo_load,
        inputs=[],
        outputs=_demo_load_outputs,
    )

    if not IS_V25:
        _preset_load_outputs = [
            experimental_checkbox,
            emo_control_method,
            prompt_audio,
            emo_upload,
            emo_weight,
            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
            emo_text,
            emo_random,
            do_sample,
            top_p,
            top_k,
            temperature,
            length_penalty,
            num_beams,
            repetition_penalty,
            max_mel_tokens,
            max_text_tokens_per_segment,
        ]
        _preset_save_inputs = [
            prompt_audio,
            emo_control_method,
            emo_upload,
            emo_weight,
            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
            emo_text,
            emo_random,
            do_sample,
            top_p,
            top_k,
            temperature,
            length_penalty,
            num_beams,
            repetition_penalty,
            max_mel_tokens,
            max_text_tokens_per_segment,
        ]

        load_preset_dropdown.change(
            on_preset_load,
            inputs=[load_preset_dropdown],
            outputs=_preset_load_outputs,
        )
        save_preset_btn.click(
            open_save_preset_modal,
            inputs=_preset_save_inputs,
            outputs=[save_preset_modal, modal_preset_preview, modal_preset_name],
        )
        modal_confirm_btn.click(
            confirm_save_preset_from_modal,
            inputs=[modal_preset_name] + _preset_save_inputs,
            outputs=[save_preset_modal, load_preset_dropdown, manage_preset_dropdown],
        )
        modal_cancel_btn.click(
            close_save_preset_modal,
            inputs=[],
            outputs=[save_preset_modal],
        )
        manage_preset_dropdown.change(
            format_preset_details,
            inputs=[manage_preset_dropdown],
            outputs=[preset_details_markdown],
        )
        manage_preset_dropdown.change(
            update_delete_preset_button,
            inputs=[manage_preset_dropdown],
            outputs=[delete_preset_btn],
        )
        apply_preset_btn.click(
            on_preset_load,
            inputs=[manage_preset_dropdown],
            outputs=_preset_load_outputs,
        )
        delete_preset_btn.click(
            on_preset_delete,
            inputs=[manage_preset_dropdown],
            outputs=[load_preset_dropdown, manage_preset_dropdown],
        )
        refresh_preset_btn.click(
            refresh_preset_choices,
            inputs=[],
            outputs=[load_preset_dropdown, manage_preset_dropdown],
        )
        create_preset_btn.click(
            on_preset_save,
            inputs=[create_preset_name] + _preset_save_inputs,
            outputs=[load_preset_dropdown, manage_preset_dropdown],
        )

    gen_button.click(gen_single,
                     inputs=[emo_control_method, prompt_audio, selected_voicepack, input_text_single,
                            lang_dropdown,
                            emo_upload, emo_weight,
                            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                             emo_text,emo_random,
                             max_text_tokens_per_segment,
                             duration_factor,
                             *advanced_params,
                     ],
                     outputs=[output_audio])



if __name__ == "__main__":
    demo.queue(20)
    demo.launch(
        server_name=cmd_args.host,
        server_port=cmd_args.port,
        allowed_paths=[VOICEPACK_EXPORT_DIR],
    )
