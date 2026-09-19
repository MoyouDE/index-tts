"""Run two opt-in experiments serially, with immutable inputs and localized artifacts.

Run with index-tts/.venv/Scripts/python.exe. Re-running resumes the same experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
for key, value in {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                   'HF_ENDPOINT': 'https://hf-mirror.com', 'TOKENIZERS_PARALLELISM': 'false',
                   'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}.items():
    os.environ[key] = value
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

DATA = REPO / 'data/emotion/dialogue-stage-20260917-deepseek-v3-final-41763'
BASELINE = REPO / 'outputs/emotion-data/macbert-training-dialogue-stage-20260917-deepseek-v3-final-41763'
DEFAULT_ROOT = Path('E:/Projects/Readest-temp/emotion-imbalance-experiment-20260919')
RUNS = ('baseline', 'A-loss', 'B-loss-sampling')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inventory(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True, encoding='utf-8').strip()


def config_for(group):
    from indextts.emotion.imbalance import ImbalanceConfig
    return ImbalanceConfig(sampling_mode='mixed' if group == 'B-loss-sampling' else 'uniform')


def run_dir(root, group):
    return root / 'baseline' if group == 'baseline' else root / 'runs' / group


def checkpoint(root, group):
    return BASELINE / 'best' if group == 'baseline' else run_dir(root, group) / 'final'


def check_inputs(root):
    manifest = read(root / 'experiment.json')
    for item in [*manifest['inputs'].values(), *manifest['baseModelFiles'].values(),
                 *manifest['baselineFiles'].values(), *manifest['sourceFiles'].values()]:
        if not Path(item['path']).is_file() or digest(item['path']) != item['sha256']:
            raise RuntimeError(f"Frozen input changed: {item['path']}")
    for package, expected in manifest['environment'].items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f'Environment changed: {package}')
    return manifest


def resources(root):
    # The coordinator must not retain a CUDA context while its training child runs.
    line = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=name,memory.free,memory.total',
                                    '--format=csv,noheader,nounits'], text=True).strip()
    gpu, free_mib, total_mib = [part.strip() for part in line.split(',')]
    free, total = int(free_mib) * 1024**2, int(total_mib) * 1024**2
    disk = shutil.disk_usage(root).free
    if free < 8 * 1024**3 or disk < 20 * 1024**3:
        raise RuntimeError(f'Resources insufficient: GPU free={free/1024**3:.2f} GiB, disk={disk/1024**3:.1f} GiB; require 8/20')
    return {'gpu': gpu, 'freeVramBytes': free, 'totalVramBytes': total, 'freeDiskBytes': disk}


def initialize(root):
    if (root / 'experiment.json').exists():
        return check_inputs(root)
    from transformers.utils.hub import cached_file
    if read(BASELINE / 'training-report.json')['bestEpoch'] != 8:
        raise RuntimeError('Baseline is not epoch 8')
    completion = read(BASELINE / 'training-complete.json')
    if digest(BASELINE / 'best/model.safetensors') != completion['bestModelSha256']:
        raise RuntimeError('Baseline completion hash mismatch')
    base_files = {}
    for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.txt', 'model.safetensors']:
        p = cached_file('hfl/chinese-macbert-base', name, local_files_only=True, _raise_exceptions_for_missing_entries=False)
        if p:
            base_files[name] = inventory(p)
    if 'model.safetensors' not in base_files:
        p = cached_file('hfl/chinese-macbert-base', 'pytorch_model.bin', local_files_only=True)
        base_files['pytorch_model.bin'] = inventory(p)
    source_paths = list((REPO / 'indextts/emotion').glob('*.py')) + [Path(__file__).resolve()]
    manifest = {
        'schema': 'readest-emotion-imbalance-experiment-v1', 'createdAt': now(),
        'inputs': {split: inventory(DATA / (split + '.jsonl')) for split in ['train', 'dev', 'test']},
        'baseModelFiles': base_files,
        'baselineFiles': {p.name: inventory(p) for p in (BASELINE / 'best').iterdir() if p.is_file()},
        'sourceFiles': {str(p.relative_to(REPO)): inventory(p) for p in source_paths},
        'gitHead': git('rev-parse', 'HEAD'), 'gitStatus': git('status', '--short'),
        'environment': {k: importlib.metadata.version(k) for k in ['torch', 'transformers', 'numpy', 'safetensors']},
        'resourceRequirements': {'minimumFreeVramGiB': 8, 'minimumFreeDiskGiB': 20,
                                 'fullLengthGpuProbeRequired': True, 'resumeTolerance': {'atol': 1e-7, 'rtol': 1e-6}},
        'training': {'epochs': 8, 'batchSize': 6, 'gradientAccumulation': 4, 'seed': 20260829,
                     'encoderLR': 2e-5, 'headLR': 1e-4, 'intensityWeight': .7, 'neutralWeight': 3., 'maxLength': 512},
        'groups': {g: config_for(g).as_dict() for g in RUNS[1:]},
        'selection': {'primaryCheckpoint': 'epoch-8', 'emotionThreshold': .35,
                      'sadMelancholicF1MinGain': .02, 'macroNonzeroMaeMinReduction': .005,
                      'maxMacroF1Drop': .01, 'maxMacroSpearmanDrop': .01, 'maxZeroFalseActivationIncrease': .01},
    }
    write(root / 'experiment.json', manifest)
    (root / 'baseline').mkdir(exist_ok=True)
    for name in ['training-report.json', 'training-complete.json', 'test-metrics.json', 'threshold-calibration.json', 'preflight.json']:
        shutil.copy2(BASELINE / name, root / 'baseline' / ('original-' + name))
    return manifest


def load_split(split):
    from indextts.emotion.schema import load_jsonl_examples
    return load_jsonl_examples(DATA / (split + '.jsonl'))


def train_options(root, group):
    manifest = read(root / 'experiment.json')
    return dict(epochs=8, batch_size=6, gradient_accumulation=4, learning_rate=2e-5,
                head_learning_rate=1e-4, intensity_loss_weight=.7, neutral_loss_weight=3.,
                max_length=512, seed=20260829, device_name='cuda', resume='auto',
                checkpoint_steps=200, keep_checkpoints=2, progress=True,
                training_config=config_for(group),
                input_manifest={'files': manifest['inputs'], 'sourceFiles': manifest['sourceFiles'],
                                'environment': manifest['environment'], 'baseModelFiles': manifest['baseModelFiles']})


def train(root, group):
    from indextts.emotion.train import train_supervised
    write(run_dir(root, group) / 'resources.json', resources(root))
    train_supervised(load_split('train'), load_split('dev'), run_dir(root, group), **train_options(root, group))


def smoke(root):
    import gc
    import torch
    from safetensors.torch import load_file
    from indextts.emotion.train import train_supervised, TrainingInterrupted
    from indextts.emotion.model import MacBertEmotionModel, masked_emotion_loss
    # Full-length synthetic stress test; no labels or data files are changed.
    resources(root)
    torch.cuda.reset_peak_memory_stats()
    probe_model = MacBertEmotionModel.from_pretrained().to('cuda')
    optimizer = torch.optim.AdamW(probe_model.parameters())
    ids = torch.ones((6, 512), dtype=torch.long, device='cuda')
    mask, types = torch.ones_like(ids), torch.zeros_like(ids)
    labels = torch.full((6, 8), .5, device='cuda')
    for _ in range(2):
        output = probe_model(ids, mask, types)
        loss, _ = masked_emotion_loss(output, labels, torch.ones_like(labels), labels.max(1, keepdim=True).values,
                                      dimension_weights=torch.ones(8, device='cuda'), regression_weight=1.,
                                      intensity_weight=.7, neutral_loss_weight=3.)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    train_peak = torch.cuda.max_memory_allocated()
    del optimizer, output, loss, ids, mask, types, labels
    gc.collect(); torch.cuda.empty_cache()
    probe_model.eval()
    with torch.inference_mode():
        ids = torch.ones((32, 512), dtype=torch.long, device='cuda')
        probe_model(ids, torch.ones_like(ids), torch.zeros_like(ids))
    torch.cuda.synchronize()
    probe = {'trainShape': [6, 512], 'inferenceShape': [32, 512], 'trainPeakAllocatedBytes': train_peak,
             'overallPeakAllocatedBytes': torch.cuda.max_memory_allocated(),
             'overallPeakReservedBytes': torch.cuda.max_memory_reserved(), 'freeBytesAfterInference': torch.cuda.mem_get_info()[0]}
    write(root / 'smoke/full-length-memory.json', probe)
    del probe_model, ids
    gc.collect(); torch.cuda.empty_cache()
    train_rows, dev_rows = load_split('train')[:48], load_split('dev')[:8]
    reports = {}
    for group in RUNS[1:]:
        options = train_options(root, group)
        options.update(epochs=1, checkpoint_steps=1, progress=False)
        destination = root / 'smoke' / group
        if (destination / 'passed.json').exists():
            reports[group] = read(destination / 'passed.json')
            continue
        full = train_supervised(train_rows, dev_rows, destination / 'continuous', **options)
        gc.collect(); torch.cuda.empty_cache()
        try:
            train_supervised(train_rows, dev_rows, destination / 'resumed', stop_after_steps=1, **options)
        except TrainingInterrupted:
            pass
        gc.collect(); torch.cuda.empty_cache()
        resumed = train_supervised(train_rows, dev_rows, destination / 'resumed', **options)
        gc.collect(); torch.cuda.empty_cache()
        x = load_file(destination / 'continuous/final/model.safetensors')
        y = load_file(destination / 'resumed/final/model.safetensors')
        error = max(float((x[k] - y[k]).abs().max()) for k in x)
        for key in x:
            torch.testing.assert_close(x[key], y[key], atol=1e-7, rtol=1e-6, msg=lambda msg: f'{group}/{key}: {msg}')
        if full['globalStep'] != resumed['globalStep']:
            raise RuntimeError(f'Resume mismatch: {group}: {error}')
        del x, y
        reports[group] = {'maxParameterError': error, 'atol': 1e-7, 'rtol': 1e-6,
                          'steps': full['globalStep'], 'passed': True}
        write(destination / 'passed.json', reports[group])
    write(root / 'smoke/passed.json', reports)


def predict(root, group, split):
    import numpy as np
    import torch
    from indextts.emotion.model import load_checkpoint
    from indextts.emotion.train import collect_model_predictions
    from indextts.emotion.metrics import calibrate_neutral_threshold
    from indextts.emotion.imbalance_metrics import detailed_metrics
    rows = load_split(split)
    dest = run_dir(root, group)
    cache = dest / f'{split}-predictions.npz'
    meta_path = dest / f'{split}-predictions.meta.json'
    identity = {'weights': digest(checkpoint(root, group) / 'model.safetensors'),
                'input': digest(DATA / (split + '.jsonl')), 'ids': [r.example_id for r in rows]}
    if cache.exists() and meta_path.exists() and read(meta_path).get('identity') == identity and read(meta_path).get('sha256') == digest(cache):
        with np.load(cache, allow_pickle=False) as archive:
            predictions = {k: archive[k] for k in archive.files}
    else:
        model, tokenizer = load_checkpoint(checkpoint(root, group))
        model.to('cuda')
        predictions = collect_model_predictions(model, rows, tokenizer, device=torch.device('cuda'),
                                                batch_size=32, max_length=512, progress=True)
        del model
        temp = cache.with_suffix('.tmp')
        with temp.open('wb') as f:
            np.savez_compressed(f, **predictions)
        temp.replace(cache)
        write(meta_path, {'identity': identity, 'sha256': digest(cache)})
    if split == 'dev':
        calibration = calibrate_neutral_threshold(predictions['intensities'], predictions['predictedIntensities'], minimum_active_recall=.8)
        write(dest / 'threshold-calibration.json', calibration)
    threshold = read(dest / 'threshold-calibration.json')['recommended']['threshold']
    metrics = detailed_metrics(predictions, threshold)
    metrics['byWork'] = {}
    for work in sorted({r.work_id for r in rows}):
        mask = np.array([r.work_id == work for r in rows])
        metrics['byWork'][work] = detailed_metrics({k: v[mask] for k, v in predictions.items()}, threshold)
    write(dest / f'{split}-metrics.json', metrics)
    out = dest / f'{split}-predictions.jsonl'
    temp = out.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8') as f:
        for i, row in enumerate(rows):
            f.write(json.dumps({'id': row.example_id, 'workId': row.work_id,
                                'labels': list(row.labels), 'vectors': predictions['vectors'][i].tolist(),
                                'labelMask': list(row.label_mask), 'intensity': row.intensity,
                                'predictedIntensity': float(predictions['predictedIntensities'][i].item())}, ensure_ascii=False) + '\n')
    temp.replace(out)


def compare(root, split):
    import numpy as np
    from indextts.emotion.imbalance_metrics import compare_dev
    from indextts.emotion.schema import EMOTION_NAMES
    all_metrics = {g: read(run_dir(root, g) / f'{split}-metrics.json') for g in RUNS}
    comparisons = {g: compare_dev(all_metrics['baseline'], all_metrics[g]) for g in RUNS[1:]}
    if split == 'test':
        comparisons = {g: {'deltas': value['deltas'], 'descriptiveOnly': True} for g, value in comparisons.items()}
    write(root / 'comparison' / f'{split}-comparison.json',
          {'createdAt': now(), 'selectionSplit': 'dev', 'metrics': all_metrics, 'comparisons': comparisons})
    rows = load_split(split)
    with np.load(run_dir(root, 'baseline') / f'{split}-predictions.npz') as f:
        baseline, labels = f['vectors'], f['labels']
    cases = {}
    for group in RUNS[1:]:
        with np.load(run_dir(root, group) / f'{split}-predictions.npz') as f:
            candidate = f['vectors']
        delta = np.abs(candidate-labels).mean(axis=1) - np.abs(baseline-labels).mean(axis=1)
        order = sorted(range(len(rows)), key=lambda i: (float(delta[i]), rows[i].example_id))
        cases[group] = {}
        for category, indices in [('improved', [i for i in order if delta[i] < 0][:10]),
                                  ('regressed', [i for i in reversed(order) if delta[i] > 0][:10])]:
            cases[group][category] = [{'id': rows[i].example_id, 'workId': rows[i].work_id,
                                      'text': rows[i].text, 'context': [s.as_json() for s in rows[i].context_sentences],
                                      'labels': dict(zip(EMOTION_NAMES, labels[i].tolist())),
                                      'baseline': dict(zip(EMOTION_NAMES, baseline[i].tolist())),
                                      'candidate': dict(zip(EMOTION_NAMES, candidate[i].tolist())),
                                      'maeDelta': float(delta[i])} for i in indices]
    write(root / 'comparison' / f'{split}-cases.json', cases)
    lines = [f'# {split} 两组初筛对比', '', '主比较固定第 8 轮。单随机种子；test 不用于调参或重新选方案。', '',
             '|组别|Macro F1|Spearman|非零宏MAE|零维误激活率|辅助维度MAE|neutral误激活/样本数|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for g, m in all_metrics.items():
        lines.append(f"|{g}|{m['macroF1']:.4f}|{m['macroSpearman']:.4f}|{m['macroNonzeroMae']:.4f}|{m['zeroDimensionFalseActivationRate']:.4f}|{m['auxiliary']['mae']:.4f}|{m['neutral']['falseActivationCount']}/{m['neutral']['count']}|")
    lines += ['', '## dev 初筛结论', '']
    decisions = comparisons if split == 'dev' else read(root / 'comparison/dev-comparison.json')['comparisons']
    for g, decision in decisions.items():
        lines.append(f"- {g}：{'值得继续验证' if decision['worthContinuing'] else '未满足全部初筛标准'}；目标改善={decision['improvesTarget']}，总体保持={decision['retainsOverall']}。")
        d = decision['deltas']
        lines.append(f"  dev 相对基线：sad/melancholic 平均 F1 {d['sadMelancholicF1']:+.4f}，非零宏 MAE {d['macroNonzeroMae']:+.4f}，Macro F1 {d['macroF1']:+.4f}，Spearman {d['macroSpearman']:+.4f}，零维误激活率 {d['zeroDimensionFalseActivationRate']:+.4f}。")
    lines += ['', 'neutral 的 dev 样本仅 7 条，阈值及误激活比例易受个别样本影响，不宜过度解读。',
              '', '## 运行成本与训练曲线', '', '|组别|训练记录耗时（小时）|主比较权重大小（MiB）|最佳轮次（仅补充）|', '|---|---:|---:|---:|']
    for g in RUNS:
        report = read(run_dir(root, g) / ('original-training-report.json' if g == 'baseline' else 'training-report.json'))
        marker = root / 'stages' / f'train-{g}.json'
        seconds = read(marker)['elapsedSeconds'] if marker.exists() else report['elapsedSeconds']
        lines.append(f"|{g}|{seconds/3600:.3f}|{(checkpoint(root, g)/'model.safetensors').stat().st_size/1024**2:.1f}|{report['bestEpoch']}|")
    write(root / 'comparison/training-curves.json', {g: read(run_dir(root, g) / ('original-training-report.json' if g == 'baseline' else 'training-report.json'))['history'] for g in RUNS})
    lines += ['', '## 各情绪指标', '', '|组别/维度|Precision|Recall|F1|Spearman|非零MAE|', '|---|---:|---:|---:|---:|---:|']
    for g, m in all_metrics.items():
        for name, v in m['perEmotion'].items():
            lines.append(f"|{g}/{name}|{v['precision']:.4f}|{v['recall']:.4f}|{v['f1']:.4f}|{v['spearman']:.4f}|{v['nonzero']['mae']:.4f}|")
    lines += ['', '详细强度分桶、有符号误差、作品指标见同名 JSON。差异案例按八维平均绝对误差改善/退化排序，各取10条，未人工筛选。',
              '', '采样曝光见各组 sampling/；训练曲线见 training-report.json。当前交接模型保持原样。']
    (root / 'comparison' / f'{split}-report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def cleanup_inventory(root):
    entries = []
    for child in root.iterdir():
        files = list(child.rglob('*')) if child.is_dir() else [child]
        entries.append({'path': str(child), 'bytes': sum(p.stat().st_size for p in files if p.is_file()),
                        'role': 'experiment-only; remove only after user selects a model'})
    write(root / 'comparison/cleanup-inventory.json', entries)
    (root / 'comparison/cleanup.md').write_text(
        '# 清理清单\n\n所有本轮产物均在实验根目录。选定方案后，先另存所选 final/、配置和报告，再整体移走实验根目录即可。\n'
        '源数据、现有基线模型、基础模型缓存和 index-tts-package 均位于目录外。\n\n' +
        '\n'.join(f"- {e['path']}：{e['bytes']/1024**3:.3f} GiB" for e in entries) + '\n', encoding='utf-8')


def orchestrate(root):
    from indextts.emotion.training_state import TrainingRunLock
    root.mkdir(parents=True, exist_ok=True)
    with TrainingRunLock(root):
        status = {'state': 'running', 'pid': os.getpid(), 'startedAt': now(), 'stage': 'initialize'}
        write(root / 'status.json', status)
        try:
            initialize(root)
            resources(root)
            stages = [('smoke', '')] + [('train', g) for g in RUNS[1:]] + [('dev', g) for g in RUNS] + [('compare-dev', '')] + [('test', g) for g in RUNS] + [('compare-test', '')]
            (root / 'logs').mkdir(exist_ok=True)
            for phase, group in stages:
                check_inputs(root)
                stage = phase + ('-' + group if group else '')
                marker = root / 'stages' / (stage + '.json')
                if marker.exists():
                    for item in read(marker)['outputs']:
                        if not Path(item['path']).is_file() or digest(item['path']) != item['sha256']:
                            raise RuntimeError('Completed stage artifact changed: ' + item['path'])
                    continue
                if phase in {'smoke', 'train', 'dev', 'test'}:
                    resources(root)
                log_path = root / 'logs' / (stage + '.log')
                stage_started = time.monotonic()
                status.update(stage=stage, updatedAt=now(), log=str(log_path))
                with log_path.open('a', encoding='utf-8') as log:
                    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--root', str(root), '--phase', phase]
                    if group:
                        command += ['--group', group]
                    child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
                    status['childPid'] = child.pid
                    write(root / 'status.json', status)
                    while child.poll() is None:
                        time.sleep(15)
                        status['updatedAt'] = now()
                        write(root / 'status.json', status)
                    if child.returncode:
                        raise RuntimeError(f'{stage} failed: exit={child.returncode}; see {log_path}')
                if phase == 'smoke':
                    outputs = [root / 'smoke/passed.json']
                elif phase == 'train':
                    dest = run_dir(root, group)
                    outputs = [dest / 'training-report.json', dest / 'training-complete.json'] + list((dest / 'final').glob('*'))
                elif phase in {'dev', 'test'}:
                    dest = run_dir(root, group)
                    outputs = [dest / f'{phase}-metrics.json', dest / f'{phase}-predictions.npz', dest / f'{phase}-predictions.meta.json', dest / f'{phase}-predictions.jsonl']
                    if phase == 'dev':
                        outputs += [dest / 'threshold-calibration.json']
                else:
                    split = phase.split('-')[1]
                    outputs = [root / 'comparison' / f'{split}-comparison.json', root / 'comparison' / f'{split}-report.md', root / 'comparison' / f'{split}-cases.json']
                write(marker, {'completedAt': now(), 'elapsedSeconds': time.monotonic() - stage_started,
                               'outputs': [inventory(p) for p in outputs]})
            check_inputs(root)
            cleanup_inventory(root)
            status.update(state='completed', stage='done', childPid=None, completedAt=now())
        except BaseException as exc:
            status.update(state='failed', error=str(exc), traceback=traceback.format_exc(), updatedAt=now())
            write(root / 'status.json', status)
            raise
        write(root / 'status.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--phase', choices=['all', 'smoke', 'train', 'dev', 'test', 'compare-dev', 'compare-test'], default='all')
    parser.add_argument('--group', choices=RUNS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == 'all':
        orchestrate(root)
    else:
        check_inputs(root)
        if args.phase == 'smoke':
            smoke(root)
        elif args.phase == 'train':
            if args.group not in RUNS[1:]:
                raise ValueError('Only A/B may be trained')
            train(root, args.group)
        elif args.phase in {'dev', 'test'}:
            if args.group is None:
                raise ValueError('Missing group')
            if args.phase == 'test' and not (root / 'comparison/dev-comparison.json').exists():
                raise RuntimeError('Freeze dev comparison before test evaluation')
            predict(root, args.group, args.phase)
        else:
            compare(root, args.phase.split('-')[1])
