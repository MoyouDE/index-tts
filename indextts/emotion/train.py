"""Supervised and domain-adaptive training loops for the MacBERT emotion model."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup

from .dataset import EmotionBatchCollator, EmotionDataset
from .metrics import emotion_metrics
from .model import DEFAULT_BASE_MODEL, MacBertEmotionModel, masked_emotion_loss, save_checkpoint
from .schema import COMMERCIAL_LICENSES, EMOTION_NAMES, EmotionExample, validate_work_splits


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


@torch.inference_mode()
def evaluate_model(
    model: MacBertEmotionModel,
    examples: Sequence[EmotionExample],
    tokenizer,
    *,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 256,
) -> dict[str, object]:
    model.eval()
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
    for batch in loader:
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
    return emotion_metrics(
        np.concatenate(vectors),
        np.concatenate(labels),
        np.concatenate(masks),
        np.concatenate(intensities),
        np.concatenate(predicted_intensities),
    )


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
    max_length: int = 256,
    seed: int = 20260829,
    device_name: str | None = None,
) -> dict[str, object]:
    validate_work_splits({"train": train_examples, "dev": dev_examples})
    seed_everything(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(str(base_model), use_fast=True)
    model = MacBertEmotionModel.from_pretrained(base_model).to(device)
    collator = EmotionBatchCollator(tokenizer, max_length=max_length)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        EmotionDataset(train_examples),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
    )
    positive_weights = emotion_positive_weights(train_examples).to(device)
    encoder_parameters = list(model.encoder.parameters())
    head_parameters = list(model.emotion_head.parameters()) + list(model.intensity_head.parameters())
    optimizer = AdamW(
        [
            {"params": encoder_parameters, "lr": learning_rate},
            {"params": head_parameters, "lr": head_learning_rate},
        ],
        weight_decay=weight_decay,
    )
    updates_per_epoch = max(1, (len(loader) + gradient_accumulation - 1) // gradient_accumulation)
    total_updates = max(1, updates_per_epoch * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_updates * warmup_ratio),
        num_training_steps=total_updates,
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    best_dir = target / "best"
    history: list[dict[str, object]] = []
    best_score = float("-inf")
    optimizer.zero_grad(set_to_none=True)
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        rolling = {"loss": 0.0, "emotionLoss": 0.0, "intensityLoss": 0.0}
        batch_count = 0
        for batch_index, batch in enumerate(loader, 1):
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
                positive_weights=positive_weights,
            )
            (loss / gradient_accumulation).backward()
            for key in rolling:
                rolling[key] += parts[key]
            batch_count += 1
            should_update = batch_index % gradient_accumulation == 0 or batch_index == len(loader)
            if should_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        metrics = evaluate_model(
            model,
            dev_examples,
            tokenizer,
            device=device,
            batch_size=max(batch_size, 16),
            max_length=max_length,
        )
        score = (
            float(metrics["macroF1"])
            + float(metrics["macroSpearman"])
            - float(metrics["neutralFalseActivationRate"])
        )
        epoch_result = {
            "epoch": epoch,
            "train": {key: value / max(1, batch_count) for key, value in rolling.items()},
            "dev": metrics,
            "selectionScore": score,
        }
        history.append(epoch_result)
        if score > best_score:
            best_score = score
            save_checkpoint(model, tokenizer, best_dir)
            (best_dir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    report = {
        "schemaVersion": 1,
        "baseModel": str(base_model),
        "precision": "fp32",
        "seed": seed,
        "maxLength": max_length,
        "trainExamples": len(train_examples),
        "devExamples": len(dev_examples),
        "emotionPositiveWeights": {
            name: float(value)
            for name, value in zip(EMOTION_NAMES, positive_weights.detach().cpu().tolist())
        },
        "elapsedSeconds": round(time.time() - started, 3),
        "bestSelectionScore": best_score,
        "history": history,
    }
    (target / "training-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
