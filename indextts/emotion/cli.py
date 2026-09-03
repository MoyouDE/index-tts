"""Command line entry point for the Readest MacBERT emotion model."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .schema import EmotionExample, load_jsonl_examples, validate_work_splits


DEFAULT_BASE_MODEL = "hfl/chinese-macbert-base"


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


def command_resplit_by_work(args) -> int:
    from .resplit import resplit_by_work

    _json(
        resplit_by_work(
            args.input,
            args.output,
            dev_works=args.dev_work,
            test_works=args.test_work,
            minimum_dev_positive=args.minimum_dev_positive,
            minimum_test_positive=args.minimum_test_positive,
        )
    )
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
            require_cuda=args.require_cuda,
            minimum_free_vram_bytes=round(args.minimum_free_vram_gib * 1024**3),
            minimum_free_disk_bytes=round(args.minimum_free_disk_gib * 1024**3),
            require_complete_context=args.require_complete_context,
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


def command_annotate_qwen_continuous_v3(args) -> int:
    from .qwen_continuous import annotate_qwen_continuous

    _json(
        annotate_qwen_continuous(
            args.candidates,
            args.output,
            args.model_dir,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            max_retries=args.max_attempts,
        )
    )
    return 0


def command_materialize_qwen_continuous_v3_valid(args) -> int:
    from .qwen_continuous import materialize_completed_qwen_continuous

    _json(materialize_completed_qwen_continuous(args.candidates, args.output))
    return 0


def command_prepare_continuous_v3(args) -> int:
    from .continuous_v3 import prepare_continuous_v3

    _json(
        prepare_continuous_v3(
            args.novel,
            args.brighter,
            args.adjudication_audit,
            args.output,
        )
    )
    return 0


def command_expand_continuous_v3_context(args) -> int:
    from .continuous_v3_context import expand_continuous_v3_context

    _json(
        expand_continuous_v3_context(
            args.input,
            args.source_root,
            args.output,
            base_model=args.base_model,
            max_length=args.max_length,
        )
    )
    return 0


def command_prepare_continuous_v3_reviews(args) -> int:
    from .continuous_v3 import prepare_continuous_v3_reviews

    report = prepare_continuous_v3_reviews(
        args.candidates,
        args.qwen_annotations,
        args.output,
        chunk_size=args.chunk_size,
    )
    compact = {key: value for key, value in report.items() if key != "chunks"}
    compact["chunkCount"] = len(report["chunks"])
    _json(compact)
    return 0


def command_prepare_continuous_v3_comparison_batches(args) -> int:
    from .continuous_v3 import prepare_continuous_v3_comparison_batches

    report = prepare_continuous_v3_comparison_batches(
        args.queue,
        args.output,
        batch_size=args.batch_size,
    )
    compact = {key: value for key, value in report.items() if key != "batches"}
    _json(compact)
    return 0


def command_validate_continuous_v3_reviews(args) -> int:
    from .continuous_v3 import validate_continuous_v3_reviews

    report = validate_continuous_v3_reviews(
        args.queue,
        args.reviews,
        args.pass_name,
    )
    _json({key: value for key, value in report.items() if key != "reviews"})
    return 0


def command_assemble_continuous_v3_review_chunks(args) -> int:
    from .continuous_v3 import assemble_continuous_v3_review_chunks

    _json(
        assemble_continuous_v3_review_chunks(
            args.queue,
            args.reviews_dir,
            args.output,
            pass_name=args.pass_name,
            chunk_size=args.chunk_size,
        )
    )
    return 0


def command_finalize_continuous_v3(args) -> int:
    from .continuous_v3 import finalize_continuous_v3

    expected_values = (
        args.expected_train_count,
        args.expected_dev_count,
        args.expected_test_count,
    )
    if any(value is not None for value in expected_values) and not all(
        value is not None for value in expected_values
    ):
        raise ValueError("自定义最终计数必须同时提供 train/dev/test")
    expected_counts = None
    if all(value is not None for value in expected_values):
        expected_counts = {
            "train": args.expected_train_count,
            "dev": args.expected_dev_count,
            "test": args.expected_test_count,
        }
    _json(
        finalize_continuous_v3(
            args.candidates,
            args.qwen_annotations,
            args.reviews,
            args.output,
            expected_split_counts=expected_counts,
            exclusions_path=args.exclusions,
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
    from .annotation import file_sha256
    from .brighter import load_brighter_split
    from .train import TrainingInterrupted, train_supervised

    train_examples = _load_many(args.train)
    dev_examples = _load_many(args.dev)
    if args.include_brighter:
        train_examples.extend(load_brighter_split("train"))
        dev_examples.extend(load_brighter_split("dev"))
    release_data = _release_data_check(train_examples)
    if args.release and not release_data["releaseEligible"]:
        raise ValueError(f"正式训练数据未达到发布门槛: {release_data}")
    input_manifest = {
        split: [
            {
                "path": Path(path).resolve().as_posix(),
                "sha256": file_sha256(path),
                "sizeBytes": Path(path).stat().st_size,
            }
            for path in paths
        ]
        for split, paths in (("train", args.train), ("dev", args.dev))
    }
    try:
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
            intensity_loss_weight=args.intensity_loss_weight,
            neutral_loss_weight=args.neutral_loss_weight,
            max_length=args.max_length,
            seed=args.seed,
            device_name=args.device,
            resume=args.resume,
            checkpoint_steps=args.checkpoint_steps,
            keep_checkpoints=args.keep_checkpoints,
            progress=args.progress,
            input_manifest=input_manifest,
        )
    except TrainingInterrupted as exc:
        print(f"INTERRUPTED: {exc}", file=sys.stderr, flush=True)
        return 130
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
        progress=args.progress,
        description="测试集评估",
        emotion_threshold=args.emotion_threshold,
        neutral_threshold=args.neutral_threshold,
    )
    if args.output:
        Path(args.output).write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _json(metrics)
    return 0


def command_calibrate_threshold(args) -> int:
    import torch

    from .metrics import calibrate_neutral_threshold
    from .model import load_checkpoint
    from .train import collect_model_predictions

    examples = _load_many(args.data)
    model, tokenizer = load_checkpoint(args.checkpoint)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    predictions = collect_model_predictions(
        model,
        examples,
        tokenizer,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        progress=args.progress,
        description="验证集阈值校准",
    )
    report = calibrate_neutral_threshold(
        predictions["intensities"],
        predictions["predictedIntensities"],
        minimum=args.minimum,
        maximum=args.maximum,
        step=args.step,
        minimum_active_recall=args.minimum_active_recall,
    )
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.threshold_output:
        Path(args.threshold_output).write_text(
            f"{float(report['recommended']['threshold']):g}\n",
            encoding="ascii",
        )
    _json(report)
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


def command_benchmark_qwen_annotations(args) -> int:
    from .release import benchmark_qwen_annotations

    examples = _load_many(args.data)
    report = benchmark_qwen_annotations(
        examples,
        args.annotations,
        neutral_adjust_calm=not args.keep_natural_as_calm,
    )
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

    resplit = sub.add_parser(
        "resplit-by-work",
        help="将既有小说数据按作品无泄漏地重新切分并记录覆盖与哈希",
    )
    resplit.add_argument("--input", action="append", required=True)
    resplit.add_argument("--output", required=True)
    resplit.add_argument("--dev-work", action="append", required=True)
    resplit.add_argument("--test-work", action="append", required=True)
    resplit.add_argument("--minimum-dev-positive", type=int, default=1)
    resplit.add_argument("--minimum-test-positive", type=int, default=1)
    resplit.set_defaults(func=command_resplit_by_work)

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
    preflight_training.add_argument("--require-cuda", action="store_true")
    preflight_training.add_argument("--minimum-free-vram-gib", type=float, default=0.0)
    preflight_training.add_argument("--minimum-free-disk-gib", type=float, default=0.0)
    preflight_training.add_argument(
        "--require-complete-context",
        action="store_true",
        help="要求 previousText 遵循最多三句完整上文且不发生上文截断",
    )
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

    qwen_continuous_v3 = sub.add_parser(
        "annotate-qwen-continuous-v3",
        help="使用目标句-only Qwen 严格生成 v3 连续八维原始伪标签",
    )
    qwen_continuous_v3.add_argument("--candidates", required=True)
    qwen_continuous_v3.add_argument("--output", required=True)
    qwen_continuous_v3.add_argument("--model-dir", required=True)
    qwen_continuous_v3.add_argument("--batch-size", type=int, default=8)
    qwen_continuous_v3.add_argument("--max-new-tokens", type=int, default=128)
    qwen_continuous_v3.add_argument("--max-attempts", type=int, default=3)
    qwen_continuous_v3.set_defaults(func=command_annotate_qwen_continuous_v3)

    materialize_qwen_continuous_v3_valid = sub.add_parser(
        "materialize-qwen-continuous-v3-valid",
        help="在明确接受失败项排除后物化严格成功的 Qwen v3 子集",
    )
    materialize_qwen_continuous_v3_valid.add_argument("--candidates", required=True)
    materialize_qwen_continuous_v3_valid.add_argument("--output", required=True)
    materialize_qwen_continuous_v3_valid.set_defaults(
        func=command_materialize_qwen_continuous_v3_valid
    )

    prepare_continuous_v3 = sub.add_parser(
        "prepare-continuous-v3",
        help="合并小说与 BRIGHTER 固定样本并生成 v3 连续标注候选",
    )
    prepare_continuous_v3.add_argument("--novel", action="append", required=True)
    prepare_continuous_v3.add_argument("--brighter", action="append", required=True)
    prepare_continuous_v3.add_argument("--adjudication-audit", required=True)
    prepare_continuous_v3.add_argument("--output", required=True)
    prepare_continuous_v3.set_defaults(func=command_prepare_continuous_v3)

    expand_continuous_v3_context = sub.add_parser(
        "expand-continuous-v3-context",
        help="从 r2 小说记录生成最多三个完整前句的独立训练快照",
    )
    expand_continuous_v3_context.add_argument("--input", required=True)
    expand_continuous_v3_context.add_argument("--source-root", required=True)
    expand_continuous_v3_context.add_argument("--output", required=True)
    expand_continuous_v3_context.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    expand_continuous_v3_context.add_argument("--max-length", type=int, default=256)
    expand_continuous_v3_context.set_defaults(func=command_expand_continuous_v3_context)

    prepare_continuous_v3_reviews = sub.add_parser(
        "prepare-continuous-v3-reviews",
        help="检测 v3 方向冲突并生成单轮 Codex Agent 校正分片",
    )
    prepare_continuous_v3_reviews.add_argument("--candidates", required=True)
    prepare_continuous_v3_reviews.add_argument("--qwen-annotations", required=True)
    prepare_continuous_v3_reviews.add_argument("--output", required=True)
    prepare_continuous_v3_reviews.add_argument("--chunk-size", type=int, default=50)
    prepare_continuous_v3_reviews.set_defaults(
        func=command_prepare_continuous_v3_reviews
    )

    comparison_batches = sub.add_parser(
        "prepare-continuous-v3-comparison-batches",
        help="按相近方向和强度排序，生成可在批内横向校准的 Agent 复核批次",
    )
    comparison_batches.add_argument("--queue", required=True)
    comparison_batches.add_argument("--output", required=True)
    comparison_batches.add_argument("--batch-size", type=int, default=100)
    comparison_batches.set_defaults(
        func=command_prepare_continuous_v3_comparison_batches
    )

    validate_continuous_v3_reviews = sub.add_parser(
        "validate-continuous-v3-reviews",
        help="严格校验一份 v3 Agent 连续八维复核结果",
    )
    validate_continuous_v3_reviews.add_argument("--queue", required=True)
    validate_continuous_v3_reviews.add_argument("--reviews", required=True)
    validate_continuous_v3_reviews.add_argument("--pass-name", required=True)
    validate_continuous_v3_reviews.set_defaults(
        func=command_validate_continuous_v3_reviews
    )

    assemble_continuous_v3_reviews = sub.add_parser(
        "assemble-continuous-v3-reviews",
        help="严格校验独立 Agent 的全部分片并原子汇总为单一复核文件",
    )
    assemble_continuous_v3_reviews.add_argument("--queue", required=True)
    assemble_continuous_v3_reviews.add_argument("--reviews-dir", required=True)
    assemble_continuous_v3_reviews.add_argument("--output", required=True)
    assemble_continuous_v3_reviews.add_argument("--pass-name", required=True)
    assemble_continuous_v3_reviews.add_argument("--chunk-size", type=int, default=50)
    assemble_continuous_v3_reviews.set_defaults(
        func=command_assemble_continuous_v3_review_chunks
    )

    finalize_continuous_v3 = sub.add_parser(
        "finalize-continuous-v3",
        help="从单轮 Agent 校正结果物化并完整审计 v3 train/dev/test 数据",
    )
    finalize_continuous_v3.add_argument("--candidates", required=True)
    finalize_continuous_v3.add_argument("--qwen-annotations", required=True)
    finalize_continuous_v3.add_argument(
        "--reviews", required=True, help="单轮 Codex Agent 汇总复核文件"
    )
    finalize_continuous_v3.add_argument("--output", required=True)
    finalize_continuous_v3.add_argument("--exclusions")
    finalize_continuous_v3.add_argument("--expected-train-count", type=int)
    finalize_continuous_v3.add_argument("--expected-dev-count", type=int)
    finalize_continuous_v3.add_argument("--expected-test-count", type=int)
    finalize_continuous_v3.set_defaults(func=command_finalize_continuous_v3)

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
    train.add_argument("--intensity-loss-weight", type=float, default=0.35)
    train.add_argument("--neutral-loss-weight", type=float, default=1.0)
    train.add_argument("--max-length", type=int, default=256)
    train.add_argument("--seed", type=int, default=20260829)
    train.add_argument("--device")
    train.add_argument("--resume", choices=["auto", "never"], default="never")
    train.add_argument("--checkpoint-steps", type=int, default=0)
    train.add_argument("--keep-checkpoints", type=int, default=2)
    train.add_argument("--progress", action="store_true")
    train.set_defaults(func=command_train)

    evaluate = sub.add_parser("evaluate", help="评估八维指标和 neutral/base 误触发率")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data", action="append", default=[])
    evaluate.add_argument("--include-brighter-test", action="store_true")
    evaluate.add_argument("--output")
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--max-length", type=int, default=256)
    evaluate.add_argument("--device")
    evaluate.add_argument("--emotion-threshold", type=float, default=0.35)
    evaluate.add_argument("--neutral-threshold", type=float, default=0.15)
    evaluate.add_argument("--progress", action="store_true")
    evaluate.set_defaults(func=command_evaluate)

    calibrate = sub.add_parser(
        "calibrate-threshold",
        help="仅用验证集校准 neutral/base gate 阈值",
    )
    calibrate.add_argument("--checkpoint", required=True)
    calibrate.add_argument("--data", action="append", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--threshold-output")
    calibrate.add_argument("--minimum", type=float, default=0.05)
    calibrate.add_argument("--maximum", type=float, default=0.8)
    calibrate.add_argument("--step", type=float, default=0.005)
    calibrate.add_argument("--minimum-active-recall", type=float, default=0.8)
    calibrate.add_argument("--batch-size", type=int, default=32)
    calibrate.add_argument("--max-length", type=int, default=256)
    calibrate.add_argument("--device")
    calibrate.add_argument("--progress", action="store_true")
    calibrate.set_defaults(func=command_calibrate_threshold)

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

    qwen_annotations = sub.add_parser(
        "benchmark-qwen-annotations",
        help="校验并评估既有 Qwen 预标注，避免重新运行数千次生成",
    )
    qwen_annotations.add_argument("--annotations", action="append", required=True)
    qwen_annotations.add_argument("--data", action="append", required=True)
    qwen_annotations.add_argument("--output", required=True)
    qwen_annotations.add_argument(
        "--keep-natural-as-calm",
        action="store_true",
        help="不把 IndexTTS Qwen 的 natural/calm 默认输出对齐为 base",
    )
    qwen_annotations.set_defaults(func=command_benchmark_qwen_annotations)

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
