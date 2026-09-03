"""Dataset adapters and tokenizer-compatible batches."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset

from .brighter import load_brighter_split
from .context_policy import CONTEXT_MAX_LENGTH, select_target_context
from .schema import EmotionExample


def format_current_text(text: str, sentence_type: str) -> str:
    prefix = "[对白]" if sentence_type == "dialogue" else "[旁白]"
    return prefix + text.strip()


class EmotionDataset(Dataset):
    def __init__(self, examples: Sequence[EmotionExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EmotionExample:
        return self.examples[index]


class EmotionBatchCollator:
    def __init__(self, tokenizer, *, max_length: int = CONTEXT_MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: Iterable[EmotionExample]) -> dict[str, object]:
        items = list(examples)
        if not any(item.context_sentences for item in items):
            encoded = self.tokenizer(
                [item.previous_text for item in items],
                [format_current_text(item.text, item.sentence_type) for item in items],
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            if "token_type_ids" not in encoded:
                encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"])
            encoded["labels"] = torch.tensor(
                [item.labels for item in items], dtype=torch.float32
            )
            encoded["label_mask"] = torch.tensor(
                [item.label_mask for item in items], dtype=torch.float32
            )
            encoded["intensity"] = torch.tensor(
                [[item.intensity] for item in items], dtype=torch.float32
            )
            encoded["example_ids"] = [item.example_id for item in items]
            return encoded

        encoded_items = []
        for item in items:
            if item.context_sentences:
                sentences = [sentence.as_json() for sentence in item.context_sentences]
                selection = select_target_context(
                    sentences,
                    item.target_sentence_id,
                    lambda text: len(
                        self.tokenizer(
                            text,
                            add_special_tokens=True,
                            truncation=False,
                            padding=False,
                        )["input_ids"]
                    ),
                    max_length=self.max_length,
                )
                encoded_items.append(
                    self.tokenizer(
                        selection.rendered_text,
                        add_special_tokens=True,
                        truncation=False,
                        padding=False,
                    )
                )
            else:
                encoded_items.append(
                    self.tokenizer(
                        item.previous_text,
                        format_current_text(item.text, item.sentence_type),
                        max_length=self.max_length,
                        truncation=True,
                        padding=False,
                    )
                )
        encoded = self.tokenizer.pad(
            encoded_items,
            padding=True,
            return_tensors="pt",
        )
        if encoded["input_ids"].shape[1] > self.max_length:
            raise ValueError("情感 batch 出现超过 maxLength 的输入")
        if "token_type_ids" not in encoded:
            encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"])
        encoded["labels"] = torch.tensor([item.labels for item in items], dtype=torch.float32)
        encoded["label_mask"] = torch.tensor(
            [item.label_mask for item in items], dtype=torch.float32
        )
        encoded["intensity"] = torch.tensor(
            [[item.intensity] for item in items], dtype=torch.float32
        )
        encoded["example_ids"] = [item.example_id for item in items]
        return encoded
