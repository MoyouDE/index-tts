"""Supervised and domain-adaptive training loops for the MacBERT emotion model."""

from __future__ import annotations

import json
import random
import signal
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup

from .context_policy import CONTEXT_MAX_LENGTH, ensure_emotion_special_tokens
from .dataset import EmotionBatchCollator, EmotionDataset
from .metrics import emotion_metrics
from .model import DEFAULT_BASE_MODEL, MacBertEmotionModel, masked_emotion_loss, save_checkpoint
from .schema import COMMERCIAL_LICENSES, EMOTION_NAMES, EmotionExample, validate_work_splits
from .training_state import (
    TRAINING_COMPLETE,
    TrainingRunLock,
    build_training_fingerprint,
    capture_rng_state,
    epoch_batch_indices,
    find_resume_checkpoint,
    mark_training_complete,
    read_completed_report,
    restore_rng_state,
    restore_training_checkpoint,
    save_training_checkpoint,
)
from .imbalance import ImbalanceConfig, dimension_weights, sampling_weights, mixed_batches, exposure_report


class TrainingInterrupted(RuntimeError):
    """Raised after a safe checkpoint has been written for an interruption."""


class _DeferredInterrupt:
    def __init__(self):
        self.requested = False
        self._previous: dict[int, object] = {}

    def _handle(self, signum, frame) -> None:
        del signum, frame
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        tqdm.write("\n收到中断请求；将在最近的 optimizer 安全边界保存 checkpoint。")

    def __enter__(self) -> "_DeferredInterrupt":
        signals = [signal.SIGINT]
        if hasattr(signal, "SIGBREAK"):
            signals.append(signal.SIGBREAK)
        for item in signals:
            self._previous[item] = signal.getsignal(item)
            signal.signal(item, self._handle)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        for item, handler in self._previous.items():
            signal.signal(item, handler)


def _gpu_progress(device: torch.device) -> dict[str, str]:
    if device.type != "cuda":
        return {}
    gib = 1024**3
    return {
        "gpu": f"{torch.cuda.memory_allocated(device) / gib:.2f}G",
        "reserved": f"{torch.cuda.memory_reserved(device) / gib:.2f}G",
        "peak": f"{torch.cuda.max_memory_allocated(device) / gib:.2f}G",
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def emotion_positive_weights(
    examples: Sequence[EmotionExample],
    *,
    maximum: float = 10.0,
) -> torch.Tensor:
    """Return per-dimension weights among active, unmasked examples only."""

    positive = np.zeros(len(EMOTION_NAMES), dtype=np.float64)
    observed = np.zeros(len(EMOTION_NAMES), dtype=np.float64)
    for example in examples:
        if example.intensity <= 1e-6:
            continue
        for index, (label, mask) in enumerate(zip(example.labels, example.label_mask)):
            if mask <= 0.5:
                continue
            observed[index] += 1.0
            positive[index] += float(label > 0.0)
    negative = observed - positive
    weights = np.ones(len(EMOTION_NAMES), dtype=np.float32)
    valid = positive > 0
    weights[valid] = np.clip(negative[valid] / positive[valid], 1.0, float(maximum))
    return torch.from_numpy(weights)


def evaluate_model(
    model: MacBertEmotionModel,
    examples: Sequence[EmotionExample],
    tokenizer,
    *,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = CONTEXT_MAX_LENGTH,
    progress: bool = False,
    description: str = "验证",
    emotion_threshold: float = 0.35,
    neutral_threshold: float = 0.15,
) -> dict[str, object]:
    predictions = collect_model_predictions(
        model,
        examples,
        tokenizer,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        progress=progress,
        description=description,
    )
    return emotion_metrics(
        predictions["vectors"],
        predictions["labels"],
        predictions["labelMasks"],
        predictions["intensities"],
        predictions["predictedIntensities"],
        threshold=emotion_threshold,
        neutral_threshold=neutral_threshold,
    )


@torch.inference_mode()
def collect_model_predictions(
    model: MacBertEmotionModel,
    examples: Sequence[EmotionExample],
    tokenizer,
    *,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = CONTEXT_MAX_LENGTH,
    progress: bool = False,
    description: str = "预测",
) -> dict[str, np.ndarray]:
    model.eval()
    if any(example.context_sentences for example in examples):
        ensure_emotion_special_tokens(tokenizer)
        embedding_count = int(model.encoder.get_input_embeddings().num_embeddings)
        if len(tokenizer) > embedding_count:
            raise ValueError("checkpoint 未包含 readest-emotion-v3 special token embeddings")
    loader = DataLoader(
        EmotionDataset(examples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=EmotionBatchCollator(tokenizer, max_length=max_length),
    )
    vectors: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    intensities: list[np.ndarray] = []
    predicted_intensities: list[np.ndarray] = []
    batches = tqdm(loader, desc=description, dynamic_ncols=True, leave=True) if progress else loader
    for batch in batches:
        output = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["token_type_ids"].to(device),
        )
        vectors.append(output.emotion_vector.cpu().numpy())
        labels.append(batch["labels"].numpy())
        masks.append(batch["label_mask"].numpy())
        intensities.append(batch["intensity"].numpy())
        predicted_intensities.append(output.intensity.cpu().numpy())
    return {
        "vectors": np.concatenate(vectors),
        "labels": np.concatenate(labels),
        "labelMasks": np.concatenate(masks),
        "intensities": np.concatenate(intensities),
        "predictedIntensities": np.concatenate(predicted_intensities),
    }


def train_supervised(
    train_examples: Sequence[EmotionExample],
    dev_examples: Sequence[EmotionExample],
    output_dir: str | Path,
    *,
    base_model: str | Path = DEFAULT_BASE_MODEL,
    epochs: int = 5,
    batch_size: int = 12,
    gradient_accumulation: int = 2,
    learning_rate: float = 2e-5,
    head_learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    intensity_loss_weight: float = 0.35,
    neutral_loss_weight: float = 1.0,
    max_length: int = CONTEXT_MAX_LENGTH,
    seed: int = 20260829,
    device_name: str | None = None,
    resume: str = "never",
    checkpoint_steps: int = 0,
    keep_checkpoints: int = 2,
    progress: bool = False,
    input_manifest: Mapping[str, object] | None = None,
    training_config: ImbalanceConfig | None = None,
    stop_after_steps: int | None = None,
) -> dict[str, object]:
    training_config = training_config or ImbalanceConfig()
    if resume not in {"auto", "never"}:
        raise ValueError("resume 必须是 auto 或 never")
    if epochs <= 0 or batch_size <= 0 or gradient_accumulation <= 0:
        raise ValueError("epochs、batch_size 和 gradient_accumulation 必须为正数")
    if checkpoint_steps < 0 or keep_checkpoints <= 0:
        raise ValueError("checkpoint_steps 不能为负数，keep_checkpoints 必须为正数")
    if intensity_loss_weight <= 0 or neutral_loss_weight <= 0:
        raise ValueError("intensity_loss_weight 和 neutral_loss_weight 必须为正数")
    validate_work_splits({"train": train_examples, "dev": dev_examples})
    seed_everything(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    config = {
        "baseModel": str(base_model),
        "precision": "fp32",
        "epochs": epochs,
        "batchSize": batch_size,
        "gradientAccumulation": gradient_accumulation,
        "learningRate": learning_rate,
        "headLearningRate": head_learning_rate,
        "weightDecay": weight_decay,
        "warmupRatio": warmup_ratio,
        "intensityLossWeight": intensity_loss_weight,
        "neutralLossWeight": neutral_loss_weight,
        "maxLength": max_length,
        "seed": seed,
        "device": str(device),
    }
    config["trainingObjective"] = training_config.as_dict()
    fingerprint = build_training_fingerprint(
        train_examples,
        dev_examples,
        config,
        input_manifest,
    )

    with TrainingRunLock(target):
        completed = read_completed_report(target, fingerprint) if resume == "auto" else None
        if completed is not None:
            from .annotation import file_sha256
            final_path = target / "final" / "model.safetensors"
            if not final_path.is_file() or file_sha256(final_path) != completed.get("finalModelSha256"):
                raise ValueError("Final checkpoint missing or hash mismatch")
            if progress:
                tqdm.write("训练已经完成且配置指纹一致；跳过重复优化。")
            return completed
        if resume == "never" and (
            (target / TRAINING_COMPLETE).exists()
            or any((target / "checkpoints").glob("checkpoint-step-*"))
        ):
            raise ValueError("输出目录已有训练状态；请使用 --resume auto 或更换输出目录")

        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求 CUDA 训练，但 PyTorch 无法使用 CUDA")
        tokenizer = AutoTokenizer.from_pretrained(str(base_model), use_fast=True)
        uses_target_context = any(example.context_sentences for example in train_examples)
        if uses_target_context:
            ensure_emotion_special_tokens(tokenizer)
        model = MacBertEmotionModel.from_pretrained(base_model)
        if uses_target_context:
            model.encoder.resize_token_embeddings(len(tokenizer))
        model = model.to(device)
        collator = EmotionBatchCollator(tokenizer, max_length=max_length)
        dataset = EmotionDataset(train_examples)
        epoch_batches = epoch_batch_indices(len(dataset), batch_size, seed, 1)
        batches_per_epoch = len(epoch_batches)
        balanced_weights, weight_report = dimension_weights(train_examples)
        sample_weights, bucket_counts = sampling_weights(train_examples)
        loss_options = {"dimension_weights": balanced_weights.to(device),
                        "regression_weight": training_config.regression_weight,
                        "regression_beta": training_config.regression_beta}
        weight_report["samplingBucketCounts"] = bucket_counts
        weight_report["samplingWeightSummary"] = {
            "min": float(sample_weights.min()), "max": float(sample_weights.max()),
            "mean": float(sample_weights.mean()), "std": float(sample_weights.std(unbiased=False)),
            "cappedFraction": float((sample_weights == 3).double().mean()),
        }
        (target / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        (target / "weights.json").write_text(json.dumps(weight_report, indent=2) + "\n", encoding="utf-8")
        encoder_parameters = list(model.encoder.parameters())
        head_parameters = list(model.emotion_head.parameters()) + list(
            model.intensity_head.parameters()
        )
        optimizer = AdamW(
            [
                {"params": encoder_parameters, "lr": learning_rate},
                {"params": head_parameters, "lr": head_learning_rate},
            ],
            weight_decay=weight_decay,
        )
        updates_per_epoch = max(
            1, (batches_per_epoch + gradient_accumulation - 1) // gradient_accumulation
        )
        total_updates = max(1, updates_per_epoch * epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=round(total_updates * warmup_ratio),
            num_training_steps=total_updates,
        )
        optimizer.zero_grad(set_to_none=True)
        best_dir = target / "best"
        history: list[dict[str, object]] = []
        best_score = float("-inf")
        global_step = 0
        resume_epoch = 1
        resume_batch = 0
        resume_rolling = {"loss": 0.0, "emotionLoss": 0.0, "intensityLoss": 0.0}
        if training_config.regression_weight:
            resume_rolling["regressionLoss"] = 0.0
        resume_batch_count = 0

        checkpoint = find_resume_checkpoint(target, fingerprint) if resume == "auto" else None
        if checkpoint is not None:
            restored = restore_training_checkpoint(
                checkpoint[0],
                expected_fingerprint=fingerprint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )
            restore_rng_state(restored.rng_state)
            history = restored.history
            best_score = restored.best_score
            global_step = restored.global_step
            resume_epoch = restored.epoch
            resume_batch = restored.batch_index
            resume_rolling.update(restored.epoch_rolling)
            resume_batch_count = restored.epoch_batch_count
            if history and int(history[-1].get("epoch", 0)) == resume_epoch:
                resume_epoch += 1
                resume_batch = 0
                resume_rolling = {key: 0.0 for key in resume_rolling}
                resume_batch_count = 0
            if progress:
                tqdm.write(
                    f"恢复 checkpoint: {restored.checkpoint_dir.name} "
                    f"(epoch={resume_epoch}, batch={resume_batch}, step={global_step})"
                )

        started = time.time()
        safe_epoch = resume_epoch
        safe_batch = resume_batch
        safe_rolling = dict(resume_rolling)
        safe_batch_count = resume_batch_count
        safe_rng = capture_rng_state()

        def persist_checkpoint() -> Path:
            return save_training_checkpoint(
                target,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                rng_state=safe_rng,
                fingerprint=fingerprint,
                epoch=safe_epoch,
                batch_index=safe_batch,
                global_step=global_step,
                history=history,
                best_score=best_score,
                epoch_rolling=safe_rolling,
                epoch_batch_count=safe_batch_count,
                keep_checkpoints=keep_checkpoints,
            )

        interrupt = _DeferredInterrupt()
        try:
            with interrupt:
                for epoch in range(resume_epoch, epochs + 1):
                    model.train()
                    all_batches = epoch_batch_indices(len(dataset), batch_size, seed, epoch)
                    if training_config.sampling_mode == "mixed":
                        all_batches = mixed_batches(sample_weights, batch_size, seed, epoch,
                                                    training_config.weighted_fraction)
                    exposures = target / "sampling"
                    exposures.mkdir(exist_ok=True)
                    (exposures / f"epoch-{epoch}.json").write_text(
                        json.dumps(exposure_report(train_examples, all_batches), indent=2) + "\n", encoding="utf-8")
                    start_batch = resume_batch if epoch == resume_epoch else 0
                    rolling = (
                        dict(resume_rolling)
                        if epoch == resume_epoch
                        else {key: 0.0 for key in resume_rolling}
                    )
                    batch_count = resume_batch_count if epoch == resume_epoch else 0
                    remaining = all_batches[start_batch:]
                    loader = DataLoader(
                        dataset,
                        batch_sampler=remaining,
                        collate_fn=collator,
                    )
                    batches = (
                        tqdm(
                            loader,
                            total=len(all_batches),
                            initial=start_batch,
                            desc=f"训练 {epoch}/{epochs}",
                            dynamic_ncols=True,
                            leave=True,
                        )
                        if progress
                        else loader
                    )
                    for relative_index, batch in enumerate(batches, 1):
                        batch_index = start_batch + relative_index
                        output = model(
                            batch["input_ids"].to(device),
                            batch["attention_mask"].to(device),
                            batch["token_type_ids"].to(device),
                        )
                        loss, parts = masked_emotion_loss(
                            output,
                            batch["labels"].to(device),
                            batch["label_mask"].to(device),
                            batch["intensity"].to(device),
                            intensity_weight=intensity_loss_weight,
                            neutral_loss_weight=neutral_loss_weight,
                            **loss_options,
                        )
                        (loss / gradient_accumulation).backward()
                        for key in rolling:
                            rolling[key] += parts[key]
                        batch_count += 1
                        should_update = (
                            batch_index % gradient_accumulation == 0
                            or batch_index == len(all_batches)
                        )
                        if should_update:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                            global_step += 1
                            safe_epoch = epoch
                            safe_batch = batch_index
                            safe_rolling = dict(rolling)
                            safe_batch_count = batch_count
                            safe_rng = capture_rng_state()
                            if stop_after_steps is not None and global_step >= stop_after_steps:
                                saved = persist_checkpoint()
                                raise TrainingInterrupted(f"Requested smoke stop: {saved}")
                            if (
                                checkpoint_steps
                                and global_step % checkpoint_steps == 0
                                and batch_index < len(all_batches)
                            ):
                                saved = persist_checkpoint()
                                if progress:
                                    tqdm.write(f"已保存 checkpoint: {saved.name}")
                            if interrupt.requested and batch_index < len(all_batches):
                                saved = persist_checkpoint()
                                raise TrainingInterrupted(f"已保存中断 checkpoint: {saved}")
                        if progress:
                            postfix = {
                                "step": global_step,
                                "loss": f"{parts['loss']:.4f}",
                                "emotion": f"{parts['emotionLoss']:.4f}",
                                "intensity": f"{parts['intensityLoss']:.4f}",
                                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                                **_gpu_progress(device),
                            }
                            batches.set_postfix(postfix, refresh=True)

                    metrics = evaluate_model(
                        model,
                        dev_examples,
                        tokenizer,
                        device=device,
                        batch_size=max(batch_size, 16),
                        max_length=max_length,
                        progress=progress,
                        description=f"验证 {epoch}/{epochs}",
                    )
                    score = (
                        float(metrics["macroF1"])
                        + float(metrics["macroSpearman"])
                        - float(metrics["neutralFalseActivationRate"])
                    )
                    epoch_result = {
                        "epoch": epoch,
                        "train": {
                            key: value / max(1, batch_count) for key, value in rolling.items()
                        },
                        "dev": metrics,
                        "selectionScore": score,
                    }
                    if history and int(history[-1].get("epoch", 0)) == epoch:
                        history[-1] = epoch_result
                    else:
                        history.append(epoch_result)
                    if score > best_score:
                        best_score = score
                        save_checkpoint(model, tokenizer, best_dir)
                        (best_dir / "metrics.json").write_text(
                            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True)
                            + "\n",
                            encoding="utf-8",
                        )
                    safe_epoch = epoch
                    safe_batch = len(all_batches)
                    safe_rolling = dict(rolling)
                    safe_batch_count = batch_count
                    safe_rng = capture_rng_state()
                    saved = persist_checkpoint()
                    if progress:
                        tqdm.write(
                            "Epoch "
                            f"{epoch}: macroF1={float(metrics['macroF1']):.4f}, "
                            f"macroSpearman={float(metrics['macroSpearman']):.4f}, "
                            "neutralFalseActivationRate="
                            f"{float(metrics['neutralFalseActivationRate']):.4f}, "
                            f"selectionScore={score:.4f}, checkpoint={saved.name}"
                        )
                    if interrupt.requested:
                        raise TrainingInterrupted(f"已保存中断 checkpoint: {saved}")
                    resume_batch = 0
                    resume_rolling = {key: 0.0 for key in resume_rolling}
                    resume_batch_count = 0
        except KeyboardInterrupt as exc:
            saved = persist_checkpoint()
            raise TrainingInterrupted(f"已保存中断 checkpoint: {saved}") from exc

        save_checkpoint(model, tokenizer, target / "final")
        report = {
            "schemaVersion": 2,
            "baseModel": str(base_model),
            "precision": "fp32",
            "seed": seed,
            "maxLength": max_length,
            "intensityLossWeight": intensity_loss_weight,
            "neutralLossWeight": neutral_loss_weight,
            "trainExamples": len(train_examples),
            "devExamples": len(dev_examples),
            "inputFingerprint": fingerprint,
            "globalStep": global_step,
            "emotionPositiveWeights": None,
            "elapsedSeconds": round(time.time() - started, 3),
            "bestSelectionScore": best_score,
            "bestEpoch": max(history, key=lambda item: float(item["selectionScore"]))[
                "epoch"
            ],
            "history": history,
        }
        from .annotation import file_sha256
        report["trainingObjective"] = training_config.as_dict()
        report["dimensionWeightReport"] = weight_report
        report["finalEpoch"] = epochs
        report["finalModelSha256"] = file_sha256(target / "final" / "model.safetensors")
        (target / "training-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        mark_training_complete(
            target,
            fingerprint=fingerprint,
            global_step=global_step,
            best_score=best_score,
        )
        return report


class _TextDataset(Dataset):
    def __init__(self, texts: Iterable[str], tokenizer, max_length: int):
        self.rows = [
            tokenizer(text, truncation=True, max_length=max_length, return_special_tokens_mask=True)
            for text in texts
            if text.strip()
        ]
        if not self.rows:
            raise ValueError("领域继续预训练语料为空")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def domain_pretrain(
    corpus_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    corpus_license: str,
    base_model: str | Path = DEFAULT_BASE_MODEL,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    max_length: int = 256,
    seed: int = 20260829,
) -> dict[str, object]:
    normalized_license = corpus_license.strip().upper()
    if normalized_license not in COMMERCIAL_LICENSES:
        raise ValueError("领域语料许可证未列入明确商用白名单")
    texts: list[str] = []
    for path in corpus_paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            texts.extend(line.strip() for line in handle if line.strip())
    seed_everything(seed)
    tokenizer = AutoTokenizer.from_pretrained(str(base_model), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(base_model))
    from transformers import DataCollatorForLanguageModeling

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    loader = DataLoader(
        _TextDataset(texts, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collator,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    losses: list[float] = []
    for _epoch in range(epochs):
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            output = model(**batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(output.loss.detach()))
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    report = {
        "baseModel": str(base_model),
        "corpusLicense": normalized_license,
        "corpusFiles": [str(Path(path).resolve()) for path in corpus_paths],
        "textCount": len(texts),
        "epochs": epochs,
        "meanLoss": float(np.mean(losses)),
        "seed": seed,
    }
    (target / "domain-pretraining-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
