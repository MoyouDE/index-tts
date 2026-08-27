from pathlib import Path
import json


def test_webui_python_source_parses():
    webui_path = Path(__file__).resolve().parents[1] / "webui.py"
    source = webui_path.read_text(encoding="utf-8")

    assert compile(source, str(webui_path), "exec") is not None


def test_webui_exposes_voicepack_builder_in_a_separate_tab():
    webui_path = Path(__file__).resolve().parents[1] / "webui.py"
    source = webui_path.read_text(encoding="utf-8")

    assert 'with gr.Tab(i18n("音色包制作"))' in source
    assert 'i18n("生成并安装音色包")' in source
    assert "export_voicepack_from_webui" in source
    assert "VoicePackBuilder" in source


def test_webui_can_install_select_and_synthesize_from_voicepacks():
    webui_path = Path(__file__).resolve().parents[1] / "webui.py"
    source = webui_path.read_text(encoding="utf-8")

    assert "当前应用音色包" in source
    assert "import_voicepack_from_webui" in source
    assert "selected_voicepack" in source
    assert "selected_voicepack = gr.State(value=_initial_voicepack)" in source
    assert "selected_voicepack = gr.Dropdown(" not in source
    assert "selected_voicepack = gr.Radio(" not in source
    assert "selected_voicepack.change(" not in source
    assert "outputs=[selected_voicepack, selected_voicepack_status, voicepack_import]" in source
    assert 'label=i18n("拖入 .ivp 音色包以安装并选中"),\n                    show_label=False,' in source
    assert 'infer_kwargs["voice_conditioning"] = voice_conditioning' in source
    assert "prebuild_example_voicepacks()" in source
    assert 'pack_path = os.path.join(VOICEPACK_DIR, f"{example[0]}.ivp")' in source
    assert "快速设置（自动选择对应音色包）" in source


def test_all_locale_files_are_valid_json():
    locale_dir = Path(__file__).resolve().parents[1] / "tools" / "i18n" / "locale"
    for path in locale_dir.glob("*.json"):
        parsed = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict) and parsed, path
