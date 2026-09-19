import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_prediction_cache_and_full_report_pipeline(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'tools/emotion-data/run_imbalance_experiment.py'
    spec = importlib.util.spec_from_file_location('imbalance_runner', script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, 'DATA', tmp_path / 'data')
    monkeypatch.setattr(runner, 'BASELINE', tmp_path / 'original-baseline')
    runner.DATA.mkdir()
    labels = np.concatenate([np.eye(8) * .8, np.zeros((1, 8)), np.array([[.8, .2, 0, 0, 0, 0, 0, 0]])])
    rows = [SimpleNamespace(example_id=f'id-{i}', work_id='work', labels=y.tolist(), label_mask=[1.] * 8,
                            intensity=float(y.max()), text='target', context_sentences=()) for i, y in enumerate(labels)]
    monkeypatch.setattr(runner, 'load_split', lambda split: rows)
    calls = []
    import indextts.emotion.model as model_module
    import indextts.emotion.train as train_module
    monkeypatch.setattr(model_module, 'load_checkpoint', lambda path: (SimpleNamespace(to=lambda device: None), None))

    def collect(*args, **kwargs):
        calls.append(1)
        return dict(vectors=labels * .9, labels=labels, labelMasks=np.ones_like(labels),
                    intensities=labels.max(axis=1).reshape(-1, 1),
                    predictedIntensities=(labels.max(axis=1) * .9).reshape(-1, 1))

    monkeypatch.setattr(train_module, 'collect_model_predictions', collect)
    root = tmp_path / 'experiment'
    for group in runner.RUNS:
        dest = runner.run_dir(root, group)
        dest.mkdir(parents=True)
        ckpt = runner.checkpoint(root, group)
        ckpt.mkdir(parents=True)
        (ckpt / 'model.safetensors').write_bytes(b'test-weights')
        runner.write(dest / ('original-training-report.json' if group == 'baseline' else 'training-report.json'),
                     dict(elapsedSeconds=10, bestEpoch=8, history=[]))
    for split in ['dev', 'test']:
        (runner.DATA / f'{split}.jsonl').write_text('test\n', encoding='utf-8')
        for group in runner.RUNS:
            runner.predict(root, group, split)
            runner.predict(root, group, split)
        runner.compare(root, split)
    assert len(calls) == 6
    assert runner.read(root / 'baseline/dev-metrics.json')['sampleCount'] == 10
    dev = runner.read(root / 'comparison/dev-comparison.json')
    test = runner.read(root / 'comparison/test-comparison.json')
    assert not dev['comparisons']['A-loss']['worthContinuing']
    assert 'worthContinuing' not in test['comparisons']['A-loss']
    assert test['comparisons']['A-loss']['descriptiveOnly']
    assert (root / 'comparison/test-report.md').exists()
    assert runner.read(root / 'comparison/test-cases.json')['A-loss']['improved'] == []
