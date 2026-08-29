"""Command line entry point for the Readest MacBERT emotion model."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

DEFAULT_BASE_MODEL = "hfl/chinese-macbert-base"

from .schema import EmotionExample, load_jsonl_examples, validate_work_splits


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _load_many(paths: list[str]) -> list[EmotionExample]:
    examples: list[EmotionExample] = []
    seen: set[str] = set()
    for path in paths:
        for example in load_jsonl_examples(path):
            if example.example_id in seen:
                raise ValueError(f"跨文件重复 id={example.example_id}")
            seen.add(example.example_id)
            examples.append(example)
    return examples


def _release_data_check(examples: list[EmotionExample]) -> dict[str, object]:
    complete = [item for item in examples if all(value > 0.5 for value in item.label_mask)]
    own_all = [item for item in complete if not item.source.startswith("brighter-dataset/")]
    approved = [item for item in own_all if item.license_status == "approved"]
    qwen_selected = [
        item for item in own_all if item.adjudication_chosen_source == "qwen"
    ]
    low_confidence = [
        item for item in own_all if item.adjudication_confidence == "low"
    ]
    works = {item.work_id for item in approved}
    eligible = (
        len(approved) == len(own_all)
        and len(approved) >= 12_000
        and len(works) >= 3
        and not qwen_selected
        and not low_confidence
    )
    return {
        "releaseEligible": eligible,
        "completeEightDimensionExamples": len(complete),
        "authorizedNovelExamples": len(approved),
        "authorizedNovelWorks": len(works),
        "novelExamplesByLicenseStatus": dict(
            Counter(item.license_status for item in own_all)
        ),
        "testOnlyNovelExamples": sum(
            item.license_status == "test-only" for item in own_all
        ),
        "pendingNovelExamples": sum(
            item.license_status == "pending" for item in own_all
        ),
        "qwenSelectedNovelExamples": len(qwen_selected),
        "lowConfidenceNovelExamples": len(low_confidence),
        "requiredNovelExamples": 12_000,
    }


def command_prepare(args) -> int:
    from .brighter import export_brighter

    _json(export_brighter(args.output))
    return 0


def command_extract_speaker_candidates(args) -> int:
    from .source_extract import extract_candidates

    _json(
        extract_candidates(
            args.input_root,
            args.output,
            pool_size=args.pool_size,
            seed=args.seed,
            license_status="test-only" if args.test_only else "pending",
        )
    )
    return 0


def command_split_candidates(args) -> int:
    from .annotation import split_candidates

    _json(
        split_candidates(
            args.candidates,
            args.output,
            chunk_size=args.chunk_size,
            limit=args.limit,
        )
    )
    return 0


def command_prepare_coverage_backlog(args) -> int:
    from .coverage import build_coverage_backlog

    _json(
        build_coverage_backlog(
            args.input_root,
            args.existing_candidates,
            args.output,
            quotas={
                "disgusted": args.disgusted,
                "melancholic": args.melancholic,
            },
            license_status="test-only" if args.test_only else "pending",
        )
    )
    return 0


def command_validate_annotations(args) -> int:
    from .annotation import file_sha256, validate_annotation_file

    report = validate_annotation_file(
        args.candidates,
        args.annotations,
        expected_input_sha256=args.input_sha256,
    )
    manifest_path = Path(args.annotations).with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "readest-emotion-annotation-v1",
                "annotationFile": str(Path(args.annotations).resolve()),
                "annotationFileSha256": file_sha256(args.annotations),
                "candidateFileSha256": report["candidateFileSha256"],
                "candidateCount": report["candidateCount"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {key: value for key, value in report.items() if key != "annotations"}
    _json(report)
    return 0


def command_merge_annotations(args) -> int:
    from .annotation import merge_double_annotations

    _json(
        merge_double_annotations(
            args.candidates,
            args.annotation_a,
            args.annotation_b,
            args.output,
            license_status="test-only" if args.test_only else "pending",
        )
    )
    return 0


def command_merge_annotation_tree(args) -> int:
    from .annotation import merge_annotation_tree

    _json(
        merge_annotation_tree(
            args.chunks,
            args.annotation_a_dir,
            args.annotation_b_dir,
            args.output,
            license_status="test-only" if args.test_only else "pending",
        )
    )
    return 0


def command_resolve_reviews(args) -> int:
    from .review import resolve_review_queue

    report = resolve_review_queue(
        args.queue,
        args.decisions,
        args.output,
        license_status="test-only" if args.test_only else "pending",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_annotation_status(args) -> int:
    from .annotation import annotation_tree_status

    _json(annotation_tree_status(args.chunks, args.annotation_a_dir, args.annotation_b_dir))
    return 0


def command_finalize_reviewed(args) -> int:
    from .finalize import finalize_reviewed_dataset

    _json(
        finalize_reviewed_dataset(
            args.merged_root,
            args.output,
            candidates_path=args.candidates,
            license_id=args.license_id,
        )
    )
    return 0


def command_prepare_training_data(args) -> int:
    from .training_prep import prepare_training_data

    _json(
        prepare_training_data(
            args.base_dir,
            args.adjudication,
            args.output,
            brighter_dir=args.brighter_dir,
            expected_adjudications=args.expected_adjudications,
            base_model=args.base_model,
            seed=args.seed,
            max_length=args.max_length,
        )
    )
    return 0


def command_integrate_coverage_data(args) -> int:
    from .supplement import integrate_reviewed_supplement

    _json(
        integrate_reviewed_supplement(
            args.base_dir,
            args.merged_dir,
            args.output,
            reviewed_dir=args.reviewed_dir,
            brighter_dir=args.brighter_dir,
            minimum_positive=args.minimum_positive,
            license_id=args.license_id,
        )
    )
    return 0


def command_preflight_training(args) -> int:
    from .training_preflight import preflight_training

    _json(
        preflight_training(
            args.train,
            args.dev,
            args.test,
            args.output,
            base_model=args.base_model,
            max_length=args.max_length,
            verify_base_weights=args.verify_base_weights,
            minimum_positive=args.minimum_positive,
        )
    )
    return 0


def command_annotate_test_qwen(args) -> int:
    from .qwen_test import annotate_qwen_test

    _json(
        annotate_qwen_test(
            args.candidates,
            args.output,
            args.model_dir,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            prompt_variant=args.prompt_variant,
        )
    )
    return 0


def command_validate(args) -> int:
    splits = {name: _load_many(paths) for name, paths in (("train", args.train), ("dev", args.dev), ("test", args.test)) if paths}
    validate_work_splits(splits)
    flattened = [item for items in splits.values() for item in items]
    report = {
        "schemaVersion": 1,
        "splits": {name: len(items) for name, items in splits.items()},
        "works": {name: len({item.work_id for item in items}) for name, items in splits.items()},
        "licenses": dict(Counter(item.license_id for item in flattened)),
        "sources": dict(Counter(item.source for item in flattened)),
        "releaseData": _release_data_check(splits.get("train", [])),
    }
    _json(report)
    return 0


def command_train(args) -> int:
    from .brighter import load_brighter_split
    from .train import train_supervised

    train_examples = _load_many(args.train)
    dev_examples = _load_many(args.dev)
    if args.include_brighter:
        train_examples.extend(load_brighter_split("train"))
        dev_examples.extend(load_brighter_split("dev"))
    release_data = _release_data_check(train_examples)
    if args.release and not release_data["releaseEligible"]:
        raise ValueError(f"正式训练数据未达到发布门槛: {release_data}")
    report = train_supervised(
        train_examples,
        dev_examples,
        args.output,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        head_learning_rate=args.head_learning_rate,
        max_length=args.max_length,
        seed=args.seed,
        device_name=args.device,
    )
    report["releaseData"] = release_data
    report["releaseTraining"] = bool(args.release)
    report_path = Path(args.output) / "training-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _json(report)
    return 0


def command_evaluate(args) -> int:
    import torch

    from .brighter import load_brighter_split
    from .model import load_checkpoint
    from .train import evaluate_model

    examples = _load_many(args.data)
    if args.include_brighter_test:
        examples.extend(load_brighter_split("test"))
    model, tokenizer = load_checkpoint(args.checkpoint)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    metrics = evaluate_model(
        model,
        examples,
        tokenizer,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    if args.output:
        Path(args.output).write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _json(metrics)
    return 0


def command_domain_pretrain(args) -> int:
    from .train import domain_pretrain

    _json(
        domain_pretrain(
            args.corpus,
            args.output,
            corpus_license=args.license,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            seed=args.seed,
        )
    )
    return 0


def command_export(args) -> int:
    from .export import export_onnx

    approval = None
    if args.release_approval:
        approval = json.loads(Path(args.release_approval).read_text(encoding="utf-8"))
    _json(
        export_onnx(
            args.checkpoint,
            args.output,
            release_approval=approval,
            version=args.version,
            neutral_threshold=args.neutral_threshold,
        )
    )
    return 0


def command_verify(args) -> int:
    from .export import verify_onnx

    _json(verify_onnx(args.checkpoint, args.model_dir, tolerance=args.tolerance))
    return 0


def command_benchmark_qwen(args) -> int:
    from .release import benchmark_qwen

    examples = _load_many(args.data)
    report = benchmark_qwen(examples, args.model_dir)
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _json(report)
    return 0


def command_release_check(args) -> int:
    from .release import build_release_approval

    approval = build_release_approval(
        training_report_path=args.training_report,
        model_metrics_path=args.model_metrics,
        qwen_metrics_path=args.qwen_metrics,
        blind_test_path=args.blind_test,
        data_audit_path=args.data_audit,
    )
    Path(args.output).write_text(
        json.dumps(approval, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _json(approval)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="indextts-emotion")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-brighter", help="下载并转换 BRIGHTER 中文强度标注")
    prepare.add_argument("--output", required=True)
    prepare.set_defaults(func=command_prepare)

    extract = sub.add_parser(
        "extract-speaker-candidates",
        help="从已有发言人标注 JSONL 抽取情感候选池",
    )
    extract.add_argument("--input-root", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--pool-size", type=int, default=20_000)
    extract.add_argument("--seed", type=int, default=20260829)
    extract.add_argument("--test-only", action="store_true", help="标记为测试数据，不执行授权门槛")
    extract.set_defaults(func=command_extract_speaker_candidates)

    split = sub.add_parser("split-candidates", help="将候选池切成 agent 标注 chunk")
    split.add_argument("--candidates", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--chunk-size", type=int, default=50)
    split.add_argument("--limit", type=int)
    split.set_defaults(func=command_split_candidates)

    coverage = sub.add_parser(
        "prepare-coverage-backlog",
        help="从未入选原候选池的句子中准备厌恶/低落定向补标队列",
    )
    coverage.add_argument("--input-root", required=True)
    coverage.add_argument("--existing-candidates", required=True)
    coverage.add_argument("--output", required=True)
    coverage.add_argument("--disgusted", type=int, default=1200)
    coverage.add_argument("--melancholic", type=int, default=2400)
    coverage.add_argument("--test-only", action="store_true")
    coverage.set_defaults(func=command_prepare_coverage_backlog)

    validate_annotations = sub.add_parser("validate-annotations", help="校验一份 agent 情感标注")
    validate_annotations.add_argument("--candidates", required=True)
    validate_annotations.add_argument("--annotations", required=True)
    validate_annotations.add_argument("--input-sha256")
    validate_annotations.set_defaults(func=command_validate_annotations)

    merge_annotations = sub.add_parser("merge-annotations", help="合并两份独立 agent 情感标注")
    merge_annotations.add_argument("--candidates", required=True)
    merge_annotations.add_argument("--annotation-a", required=True)
    merge_annotations.add_argument("--annotation-b", required=True)
    merge_annotations.add_argument("--output", required=True)
    merge_annotations.add_argument("--test-only", action="store_true", help="标记为测试合并结果")
    merge_annotations.set_defaults(func=command_merge_annotations)

    merge_tree = sub.add_parser(
        "merge-annotation-tree",
        help="校验并合并所有已完成的双标 chunk（缺任何一片即拒绝）",
    )
    merge_tree.add_argument("--chunks", required=True)
    merge_tree.add_argument("--annotation-a-dir", required=True)
    merge_tree.add_argument("--annotation-b-dir", required=True)
    merge_tree.add_argument("--output", required=True)
    merge_tree.add_argument("--test-only", action="store_true", help="标记为测试合并结果")
    merge_tree.set_defaults(func=command_merge_annotation_tree)

    resolve_reviews = sub.add_parser(
        "resolve-reviews", help="校验并写出人工复核后的 review queue 结果"
    )
    resolve_reviews.add_argument("--queue", required=True)
    resolve_reviews.add_argument("--decisions", required=True)
    resolve_reviews.add_argument("--output", required=True)
    resolve_reviews.add_argument("--test-only", action="store_true", help="标记为测试复核结果")
    resolve_reviews.set_defaults(func=command_resolve_reviews)

    status = sub.add_parser(
        "annotation-status",
        help="统计双标分片完成度，不生成训练文件",
    )
    status.add_argument("--chunks", required=True)
    status.add_argument("--annotation-a-dir", required=True)
    status.add_argument("--annotation-b-dir", required=True)
    status.set_defaults(func=command_annotation_status)

    finalize = sub.add_parser(
        "finalize-reviewed-test",
        help="将全部复核 chunk 聚合为 test-only EmotionExample train/dev/test",
    )
    finalize.add_argument("--merged-root", required=True)
    finalize.add_argument("--output", required=True)
    finalize.add_argument("--candidates")
    finalize.add_argument("--license-id", default="PROPRIETARY-AUTHORIZED")
    finalize.set_defaults(func=command_finalize_reviewed)

    prepare_training = sub.add_parser(
        "prepare-training-data",
        help="应用二次决断并生成不启动训练的 test-only 训练快照",
    )
    prepare_training.add_argument("--base-dir", required=True)
    prepare_training.add_argument("--adjudication", action="append", required=True)
    prepare_training.add_argument("--output", required=True)
    prepare_training.add_argument("--brighter-dir")
    prepare_training.add_argument("--expected-adjudications", type=int)
    prepare_training.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    prepare_training.add_argument("--seed", type=int, default=20260829)
    prepare_training.add_argument("--max-length", type=int, default=256)
    prepare_training.set_defaults(func=command_prepare_training_data)

    integrate_coverage = sub.add_parser(
        "integrate-coverage-data",
        help="将完成双标和复核的覆盖样本追加到既有训练快照",
    )
    integrate_coverage.add_argument("--base-dir", required=True)
    integrate_coverage.add_argument("--merged-dir", required=True)
    integrate_coverage.add_argument("--reviewed-dir")
    integrate_coverage.add_argument("--brighter-dir")
    integrate_coverage.add_argument("--output", required=True)
    integrate_coverage.add_argument("--minimum-positive", type=int, default=1_000)
    integrate_coverage.add_argument("--license-id", default="PROPRIETARY-AUTHORIZED")
    integrate_coverage.set_defaults(func=command_integrate_coverage_data)

    preflight_training = sub.add_parser(
        "preflight-training",
        help="只读校验训练输入、MacBERT 本地缓存、长度分布和作品泄漏",
    )
    preflight_training.add_argument("--train", action="append", required=True)
    preflight_training.add_argument("--dev", action="append", required=True)
    preflight_training.add_argument("--test", action="append", required=True)
    preflight_training.add_argument("--output", required=True)
    preflight_training.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    preflight_training.add_argument("--max-length", type=int, default=256)
    preflight_training.add_argument("--verify-base-weights", action="store_true")
    preflight_training.add_argument("--minimum-positive", type=int, default=1_000)
    preflight_training.set_defaults(func=command_preflight_training)

    qwen_test = sub.add_parser(
        "annotate-test-qwen",
        help="使用本机 Qwen 生成可复核的 test-only 预标注（非人工标签）",
    )
    qwen_test.add_argument("--candidates", required=True)
    qwen_test.add_argument("--output", required=True)
    qwen_test.add_argument("--model-dir", required=True)
    qwen_test.add_argument("--batch-size", type=int, default=8)
    qwen_test.add_argument("--max-new-tokens", type=int, default=128)
    qwen_test.add_argument("--prompt-variant", choices=["a", "b"], default="a")
    qwen_test.set_defaults(func=command_annotate_test_qwen)

    validate = sub.add_parser("validate-data", help="校验 schema、许可证白名单和作品级切分")
    validate.add_argument("--train", action="append", default=[])
    validate.add_argument("--dev", action="append", default=[])
    validate.add_argument("--test", action="append", default=[])
    validate.set_defaults(func=command_validate)

    dapt = sub.add_parser("pretrain-domain", help="在权利明确的小说文本上继续 MLM 预训练")
    dapt.add_argument("--corpus", action="append", required=True)
    dapt.add_argument("--license", required=True)
    dapt.add_argument("--output", required=True)
    dapt.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    dapt.add_argument("--epochs", type=int, default=1)
    dapt.add_argument("--batch-size", type=int, default=8)
    dapt.add_argument("--learning-rate", type=float, default=5e-5)
    dapt.add_argument("--max-length", type=int, default=256)
    dapt.add_argument("--seed", type=int, default=20260829)
    dapt.set_defaults(func=command_domain_pretrain)

    train = sub.add_parser("train", help="微调八维情感与总强度双头模型")
    train.add_argument("--train", action="append", default=[])
    train.add_argument("--dev", action="append", default=[])
    train.add_argument("--include-brighter", action="store_true")
    train.add_argument("--release", action="store_true", help="强制检查 12000 条完整八维小说标注门槛")
    train.add_argument("--output", required=True)
    train.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=12)
    train.add_argument("--gradient-accumulation", type=int, default=2)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--head-learning-rate", type=float, default=1e-4)
    train.add_argument("--max-length", type=int, default=256)
    train.add_argument("--seed", type=int, default=20260829)
    train.add_argument("--device")
    train.set_defaults(func=command_train)

    evaluate = sub.add_parser("evaluate", help="评估八维指标和 neutral/base 误触发率")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data", action="append", default=[])
    evaluate.add_argument("--include-brighter-test", action="store_true")
    evaluate.add_argument("--output")
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--max-length", type=int, default=256)
    evaluate.add_argument("--device")
    evaluate.set_defaults(func=command_evaluate)

    export = sub.add_parser("export-onnx", help="导出 FP32 ONNX 与哈希 manifest")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--release-approval", help="完整发布门槛均通过且含证据哈希的 JSON 文件")
    export.add_argument("--version", default="1.0.0")
    export.add_argument("--neutral-threshold", type=float, default=0.15)
    export.set_defaults(func=command_export)

    verify = sub.add_parser("verify-onnx", help="对比 PyTorch/ONNX 数值并校验文件哈希")
    verify.add_argument("--checkpoint", required=True)
    verify.add_argument("--model-dir", required=True)
    verify.add_argument("--tolerance", type=float, default=1e-4)
    verify.set_defaults(func=command_verify)

    qwen = sub.add_parser("benchmark-qwen", help="在同一人工测试集上记录旧 Qwen 基线")
    qwen.add_argument("--model-dir", required=True)
    qwen.add_argument("--data", action="append", required=True)
    qwen.add_argument("--output", required=True)
    qwen.set_defaults(func=command_benchmark_qwen)

    release = sub.add_parser("release-check", help="组合数据、指标和 TTS 盲测发布门槛")
    release.add_argument("--training-report", required=True)
    release.add_argument("--model-metrics", required=True)
    release.add_argument("--qwen-metrics", required=True)
    release.add_argument("--blind-test", required=True)
    release.add_argument("--data-audit", required=True)
    release.add_argument("--output", required=True)
    release.set_defaults(func=command_release_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
