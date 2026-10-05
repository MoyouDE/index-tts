import copy
import json
import uuid
import subprocess
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from indextts.material_service import MaterialService, write_json
from indextts.material_vad import SileroVad, speech_intervals


def test_silence_setting_keeps_short_pauses_inside_continuous_speech():
    probabilities = [.9] * 30 + [0.] * 20 + [.9] * 30
    duration = len(probabilities) * .032
    separate, raw = speech_intervals(probabilities, duration, min_silence_ms=500)
    continuous, joined = speech_intervals(probabilities, duration, min_silence_ms=1000)
    assert len(separate) == len(raw) == 2
    assert len(continuous) == len(joined) == 1
    assert joined[0] == (0., duration)


@pytest.mark.parametrize("gap", [100, 500, 3000])
def test_custom_silence_still_bounds_reference_length(gap):
    probabilities = [.9] * 1400
    segments, _ = speech_intervals(probabilities, 44.8, min_silence_ms=gap)
    assert all(0 < s["end"] - s["start"] <= 15 for s in segments)
    assert all(a["end"] <= b["start"] for a, b in zip(segments, segments[1:]))


@pytest.mark.parametrize("silence,volume", [(50, None), (3001, None),
    (float("nan"), None), (True, None), (500, -81), (500, 1),
    (500, float("nan")), (500, True)])
def test_invalid_detection_settings_rejected_before_loading_model(tmp_path, silence, volume):
    detector = SileroVad(tmp_path / "missing.onnx")
    with pytest.raises(ValueError):
        detector.detect(tmp_path / "missing.wav", min_silence_ms=silence, min_volume_db=volume)
    assert detector._model is None


def test_volume_floor_ignores_faint_noise_without_changing_samples(tmp_path):
    # A deliberately permissive detector proves that the amplitude filter is effective
    # independently of the model's speech probability.
    class Model:
        def run(self, _, inputs):
            return np.array([[.9]], dtype=np.float32), inputs["state"]

    path = tmp_path / "audio.wav"
    samples = np.concatenate([np.full(16000, .1), np.full(16000, .001), np.full(16000, .1)])
    sf.write(path, samples, 16000, subtype="FLOAT")
    before = path.read_bytes()
    detector = SileroVad()
    detector._model = Model()
    detector._model_sha = "test"
    original = detector.detect(path)
    filtered = detector.detect(path, min_silence_ms=500, min_volume_db=-30)
    joined = detector.detect(path, min_silence_ms=1500, min_volume_db=-30)
    empty = detector.detect(path, min_volume_db=-10)
    assert len(original["speechIntervals"]) == 1
    assert len(filtered["speechIntervals"]) == 2
    assert len(joined["speechIntervals"]) == 1
    assert not empty["segments"]
    assert filtered["settings"]["minVolumeDb"] == -30
    assert joined["settings"]["minSilenceMs"] == 1500
    assert path.read_bytes() == before


def test_short_filter_preserves_ranges_and_prior_exclusions():
    from indextts.material_web import exclude_short_ranges
    values = [["noise", 0., .3, True], ["speech", 1., 4., True],
              ["excluded", 5., 8., False], ["boundary", 9., 10., True]]
    before = copy.deepcopy(values)
    result, count = exclude_short_ranges(values, 1.)
    assert count == 1
    assert result == [["noise", 0., .3, False], *before[1:]]
    assert values == before
    assert exclude_short_ranges(result, 1.) == (result, 0)


@pytest.mark.parametrize("minimum", [0, -1, 16, float("nan"), True])
def test_invalid_short_filter_does_not_change_selection(minimum):
    from indextts.material_web import exclude_short_ranges
    values = [["a", 0., 1., True]]
    with pytest.raises(ValueError):
        exclude_short_ranges(values, minimum)
    assert values == [["a", 0., 1., True]]


def test_ui_import_settings_and_selected_draft_are_separate(tmp_path, monkeypatch):
    from indextts.validation_web import create_app
    workspace = tmp_path / "work"
    service = MaterialService(workspace)
    source_id = uuid.uuid4().hex
    folder = service.directory(source_id)
    folder.mkdir()
    sf.write(folder / "audio.wav", np.zeros(16000 * 8), 16000, subtype="FLOAT")
    record = {"sourceId": source_id, "name": "source.wav", "sourceSha256": "a" * 64,
        "audioSha256": "b" * 64, "durationSeconds": 8., "speechSeconds": 7., "primary": "long",
        "segments": [{"id": "long", "start": 0., "end": 5., "selected": True}]}
    record["revision"] = service.selection_revision(record)
    write_json(folder / "material.json", record)
    app = create_app(modules="producer", workspace_dir=workspace, output_dir=tmp_path / "out")
    callbacks = {f.fn.__name__: f.fn for f in app.fns.values() if f.fn}
    captured = []

    def import_media(path, progress=None, **settings):
        captured.append((path, settings))
        return record

    monkeypatch.setattr(app.workbench.materials, "import_media", import_media)
    try:
        labels = [c.get('props', {}).get('label') for c in app.config['components']]
        assert '选中片段都是同一人，且已排除不需要的声音' not in labels
        assert '参考构建方法' not in labels
        assert any(c.get('props', {}).get('elem_id') == 'material-quality-hint' for c in app.config['components'])
        callbacks["import_material"]('source.wav', 1.2, True, -40.)
        assert captured == [('source.wav', {"min_silence_ms": 1200, "min_volume_db": -40.})]
        assert 'suggest' not in callbacks and 'filter_short' not in callbacks
        # Mirroring receives confirmed references, independently of the candidate list.
        draft = json.dumps({"sourceId": source_id,
            "rows": [["long", 0., .4, True], ["keep", 1., 5., True]], "primary": "long"})
        result = callbacks['sync_draft'](source_id, draft)
        assert len(result) == 3
        assert result[0] == json.loads(draft)['rows']
        assert result[2]['value'] == 'long'
        load = next(f for f in app.fns.values() if f.fn and f.fn.__name__ == 'load')
        assert len(load.outputs) == len(callbacks['load'](None)) == len(callbacks['load'](source_id)) == 11
        assert service.record(source_id) == record
    finally:
        app.workbench.close()


def test_parameter_changes_reuse_raw_analysis_and_persistent_cache(tmp_path):
    class Model:
        calls = 0
        def run(self, _, inputs):
            self.calls += 1
            return np.array([[.9]], dtype=np.float32), inputs['state']
    path = tmp_path/'vad.wav'
    sf.write(path, np.full(32000, .01), 16000, subtype='FLOAT')
    model = Model()
    detector = SileroVad()
    detector._model, detector._model_sha = model, 'test'
    assert detector.detect(path)['segments']
    calls = model.calls
    assert not detector.detect(path, min_volume_db=-20)['segments']
    assert detector.detect(path, min_volume_db=-60, min_silence_ms=1200)['segments']
    assert model.calls == calls
    another = SileroVad()
    another._model, another._model_sha = model, 'test'
    assert another.analyze(path)['probabilities'] == detector.analyze(path)['probabilities']
    assert model.calls == calls  # Restart/another instance reads disk, too.
    (tmp_path/'vad-analysis-v1.json').write_text('broken', encoding='utf-8')
    another._analyses.clear()
    another.analyze(path)
    assert model.calls == calls * 2
    sf.write(path, np.full(48000, .1), 16000, subtype='FLOAT')
    assert another.analyze(path)['duration'] == 3
    assert model.calls > calls * 2


def test_browser_segmentation_matches_cpu_results_and_staging_is_explicit():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for browser postprocessing parity')
    algorithm = Path(__file__).resolve().parents[1]/'indextts/material_candidates.js'
    rng = np.random.default_rng(42)
    signals = [[.9]*30+[0.]*20+[.9]*30, [.9]*1400, [0.]*100,
               rng.choice([0., .34, .35, .49, .5, .9], 2000).tolist()]
    cases, expected = [], []
    for probabilities in signals:
        rms = rng.choice([.001, .01, .1], len(probabilities)).tolist()
        duration = len(probabilities)*.032
        for gap in [.1, .5, 1.2, 3.]:
            for volume in [None, -30.]:
                for minimum in [.1, 1., 15.]:
                    cases.append([{'probabilities': probabilities, 'rms': rms, 'duration': duration}, gap, volume, minimum])
                    floor = 10**(volume/20) if volume is not None else None
                    masked = [p if floor is None or r >= floor else 0. for p, r in zip(probabilities, rms)]
                    segments, _ = speech_intervals(masked, duration, min_silence_ms=gap*1000)
                    expected.append([[s['start'], s['end']] for s in segments if s['end']-s['start'] >= minimum-1e-6])
    script = """
    const {candidateRanges, stageRange} = require(process.argv[1]);
    let input='';process.stdin.on('data',d=>input+=d);process.stdin.on('end',()=>{
      const cases=JSON.parse(input);
      const ranges=cases.map(c=>candidateRanges(...c).rows.map(r=>r.slice(1,3)));
      const original=[['kept',1,3,true]];
      const pending=['new',4,6,true];
      const confirmed=stageRange(original,pending,20);
      const adjusted=stageRange(confirmed,['new',4.5,6,true],20);
      let overlap=false,tooLong=false;
      try{stageRange(original,['bad',2,5,true],20)}catch(e){overlap=true}
      try{stageRange(original,['bad',4,20,true],20)}catch(e){tooLong=true}
      console.log(JSON.stringify({ranges,original,pending,confirmed,adjusted,overlap,tooLong}));
    });
    """
    result = subprocess.run([node, '-e', script, str(algorithm)], input=json.dumps(cases),
                            capture_output=True, text=True, check=True)
    value = json.loads(result.stdout)
    assert value['ranges'] == expected
    assert value['original'] == [['kept', 1, 3, True]]
    assert value['pending'] == ['new', 4, 6, True]
    assert value['confirmed'] == [['kept', 1, 3, True], ['new', 4, 6, True]]
    assert value['adjusted'] == [['kept', 1, 3, True], ['new', 4.5, 6, True]]
    assert value['overlap'] and value['tooLong']
