from pathlib import Path
import json


def test_webui_python_source_parses():
    webui_path = Path(__file__).resolve().parents[1] / "webui.py"
    source = webui_path.read_text(encoding="utf-8")

    assert compile(source, str(webui_path), "exec") is not None


def test_webui_exposes_voicepack_export_separately_from_presets():
    webui_path = Path(__file__).resolve().parents[1] / "webui.py"
    source = webui_path.read_text(encoding="utf-8")

    assert 'i18n("保存为预设")' in source
    assert 'i18n("导出音色包")' in source
    assert "export_voicepack_from_webui" in source
    assert "VoicePackBuilder" in source


def test_webui_can_install_select_and_synthesize_from_voicepacks():
    webui_path = Path(__file__).resolve().parents[1] / "webui.py"
    source = webui_path.read_text(encoding="utf-8")

    assert "当前应用音色包" in source
    assert "import_voicepack_from_webui" in source
    assert "selected_voicepack" in source
    assert 'infer_kwargs["voice_conditioning"] = voice_conditioning' in source
    assert "_example_voicepack(example[0])" in source
    assert "使用上方参考音频（不使用音色包）" in source


def test_all_locale_files_are_valid_json():
    locale_dir = Path(__file__).resolve().parents[1] / "tools" / "i18n" / "locale"
    for path in locale_dir.glob("*.json"):
        parsed = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict) and parsed, path
