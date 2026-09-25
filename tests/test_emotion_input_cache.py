"""Exact-input checks for the in-process emotion tokenization cache."""

from __future__ import annotations

import random

import torch
from transformers import BertTokenizerFast

from indextts.emotion.context_policy import ensure_emotion_special_tokens
from indextts.emotion.dataset import EmotionBatchCollator
from indextts.emotion.schema import EmotionContextSentence, EmotionExample
from indextts.emotion.training_preflight import (
    _complete_context_summary,
    _inspect_examples,
    _token_lengths,
)


def _tokenizer(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *"你好再见世界情绪对白旁白目标句长短甲乙丙丁。"])
        + "\n",
        encoding="utf-8",
    )
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))
    ensure_emotion_special_tokens(tokenizer)
    return tokenizer


def _example(index: int, *, long_target: bool = False, legacy: bool = False) -> EmotionExample:
    labels = (0.4, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0)
    sentences = tuple(
        EmotionContextSentence(
            sentence_id=str(number),
            section_id="book:1",
            line_index=number,
            text=("目标句。" * 80 if long_target and number == 2 else f"你好世界{index}。"),
            sentence_type="dialogue" if number == 2 else "narration",
        )
        for number in range(4)
    )
    return EmotionExample(
        example_id=f"sample-{index}",
        work_id="book",
        previous_text="你好。" if legacy else "",
        text=sentences[2].text,
        sentence_type="dialogue",
        labels=labels,
        label_mask=(1.0,) * 8,
        intensity=0.4,
        license_id="PROPRIETARY-AUTHORIZED",
        source="test",
        context_sentences=() if legacy else sentences,
        target_sentence_id=None if legacy else "2",
        section_id=None if legacy else "book:1",
    )


def test_cached_context_batches_match_uncached_exactly(tmp_path, monkeypatch):
    tokenizer = _tokenizer(tmp_path)
    examples = [_example(index, long_target=index == 3) for index in range(35)]
    examples.append(_example(99, legacy=True))
    uncached = EmotionBatchCollator(tokenizer, max_length=64)
    cached = EmotionBatchCollator(tokenizer, max_length=64)
    cached.prepare_contexts(examples)
    assert len(cached._context_encodings) == 35

    def unexpected_encoding(_item):
        raise AssertionError("context re-tokenized")

    monkeypatch.setattr(cached, "_encode_context", unexpected_encoding)

    rng = random.Random(42)
    for batch_size in (1, 6, 16, 32):
        for _ in range(8):
            batch = rng.sample(examples, batch_size)
            original = uncached(batch)
            prepared = cached(batch)
            assert original.keys() == prepared.keys()
            for key in original:
                if isinstance(original[key], torch.Tensor):
                    assert torch.equal(original[key], prepared[key]), key
                else:
                    assert original[key] == prepared[key], key


def test_single_pass_preflight_matches_separate_scans(tmp_path):
    tokenizer = _tokenizer(tmp_path)
    examples = [_example(0), _example(1, long_target=True), _example(2, legacy=True)]

    class CountingTokenizer:
        def __init__(self):
            self.calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            return tokenizer(*args, **kwargs)

    separate = CountingTokenizer()
    expected_lengths = _token_lengths(separate, examples, max_length=64)
    expected_summary = _complete_context_summary(separate, examples, max_length=64)
    combined = CountingTokenizer()
    lengths, summary = _inspect_examples(
        combined,
        examples,
        max_length=64,
        include_lengths=True,
        include_context=True,
    )
    assert lengths == expected_lengths
    assert summary == expected_summary
    assert combined.calls < separate.calls
