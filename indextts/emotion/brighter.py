"""Pure-Python BRIGHTER adapter kept separate from the PyTorch training stack."""

from __future__ import annotations

import json
from pathlib import Path

from .schema import EMOTION_NAMES, EmotionExample


BRIGHTER_REPO = "brighter-dataset/BRIGHTER-emotion-intensities"
BRIGHTER_CONFIG = "chn"
BRIGHTER_LICENSE = "CC-BY-4.0"
BRIGHTER_MAP = {
    "joy": "happy",
    "anger": "angry",
    "sadness": "sad",
    "fear": "afraid",
    "disgust": "disgusted",
    "surprise": "surprised",
}


def load_brighter_split(split: str) -> list[EmotionExample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("读取 BRIGHTER 需要安装 emotion_train extra 或使用 emotion-data 工具环境") from exc
    rows = load_dataset(BRIGHTER_REPO, BRIGHTER_CONFIG, split=split)
    examples: list[EmotionExample] = []
    for row in rows:
        labels = [0.0] * len(EMOTION_NAMES)
        mask = [0.0] * len(EMOTION_NAMES)
        for source_name, target_name in BRIGHTER_MAP.items():
            index = EMOTION_NAMES.index(target_name)
            labels[index] = float(row[source_name]) / 3.0
            mask[index] = 1.0
        examples.append(
            EmotionExample(
                example_id=f"brighter-chn-{split}-{row['id']}",
                work_id=f"brighter-chn-{split}",
                previous_text="",
                text=str(row["text"]).strip(),
                sentence_type="narration",
                labels=tuple(labels),
                label_mask=tuple(mask),
                intensity=max(labels),
                license_id=BRIGHTER_LICENSE,
                source=f"{BRIGHTER_REPO}@chn/{split}",
            )
        )
    return examples


def export_brighter(output_dir: str | Path) -> dict[str, int]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        examples = load_brighter_split(split)
        with (target / f"brighter-chn-{split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for example in examples:
                handle.write(json.dumps(example.as_json(), ensure_ascii=False, sort_keys=True) + "\n")
        counts[split] = len(examples)
    attribution = {
        "dataset": BRIGHTER_REPO,
        "config": BRIGHTER_CONFIG,
        "license": BRIGHTER_LICENSE,
        "homepage": f"https://huggingface.co/datasets/{BRIGHTER_REPO}",
        "note": "melancholic 与 calm 未标注，训练时通过 labelMask 屏蔽。",
    }
    (target / "BRIGHTER-ATTRIBUTION.json").write_text(
        json.dumps(attribution, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return counts
