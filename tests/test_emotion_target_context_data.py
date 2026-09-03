from __future__ import annotations

import pytest

from indextts.emotion.schema import EMOTION_NAMES, TARGET_CONTEXT_SCHEMA, parse_example
from indextts.emotion.target_context_data import _materialize_record


class CharacterTokenizer:
    def __call__(self, text, **_kwargs):
        return {"input_ids": list(range(len(text) + 2))}


def record(sentence_id: int = 2):
    return {
        "schema": "readest-emotion-continuous-v3",
        "schemaVersion": 3,
        "id": f"emotion-candidate-book-r000001-s{sentence_id:06d}",
        "split": "train",
        "workId": "book",
        "previousText": "旧窗口。",
        "text": "你他妈的！",
        "sentenceType": "dialogue",
        "sourceKind": "novel",
        "source": "speaker-id:book_annotated",
        "licenseId": "PROPRIETARY-AUTHORIZED",
        "licenseStatus": "test-only",
        "emotions": {name: index / 100 for index, name in enumerate(EMOTION_NAMES)},
    }


def source_data(next_line_index: int = 1):
    values = [
        {"sentenceId": 0, "sectionId": "book:1", "chapterIndex": 1, "lineIndex": 0, "text": "更早。", "sentenceType": "narration", "sourceOffset": 0},
        {"sentenceId": 1, "sectionId": "book:1", "chapterIndex": 1, "lineIndex": 1, "text": "前句。", "sentenceType": "narration", "sourceOffset": 4},
        {"sentenceId": 2, "sectionId": "book:1", "chapterIndex": 1, "lineIndex": 1, "text": "你他妈的！", "sentenceType": "dialogue", "sourceOffset": 8},
        {"sentenceId": 3, "sectionId": "book:1", "chapterIndex": 1, "lineIndex": next_line_index, "text": "XX笑着说。", "sentenceType": "narration", "sourceOffset": 14},
    ]
    return {("book", value["sentenceId"]): value for value in values}, {"book:1": values}


def source_rows():
    return {
        ("book_annotated", 1): {
            "sentences": [
                {"sentenceId": 2, "text": "你他妈的！", "type": "dialogue"}
            ]
        }
    }


def exact_index(sentence_index):
    return {
        ("book", value["text"], value["sentenceType"]): [value]
        for value in sentence_index.values()
    }


def test_materialization_removes_previous_text_and_preserves_labels():
    sentence_index, sections = source_data()
    original = record()
    output, audit = _materialize_record(
        original,
        sentence_index,
        sections,
        exact_index(sentence_index),
        source_rows(),
        {},
        CharacterTokenizer(),
        max_length=512,
    )

    assert output["schema"] == TARGET_CONTEXT_SCHEMA
    assert "previousText" not in output
    assert "text" not in output
    assert "sentenceType" not in output
    assert output["emotions"] == original["emotions"]
    assert [value["sentenceId"] for value in output["sentences"]] == [0, 1, 2, 3]
    assert audit["nextSentenceIncluded"] is True
    parsed = parse_example(output)
    assert parsed.target_sentence_id == "2"
    assert parsed.labels == tuple(original["emotions"].values())
    assert parsed.as_json()["schema"] == TARGET_CONTEXT_SCHEMA
    assert "previousText" not in parsed.as_json()


def test_target_context_schema_rejects_legacy_context_fields():
    sentence_index, sections = source_data()
    output, _ = _materialize_record(
        record(),
        sentence_index,
        sections,
        exact_index(sentence_index),
        source_rows(),
        {},
        CharacterTokenizer(),
        max_length=512,
    )
    output["previousText"] = "不得出现"

    with pytest.raises(ValueError, match="旧上下文字段"):
        parse_example(output)


def test_post_target_sentence_is_excluded_after_a_line_break():
    sentence_index, sections = source_data(next_line_index=2)
    output, audit = _materialize_record(
        record(),
        sentence_index,
        sections,
        exact_index(sentence_index),
        source_rows(),
        {},
        CharacterTokenizer(),
        max_length=512,
    )

    assert [value["sentenceId"] for value in output["sentences"]] == [0, 1, 2]
    assert audit["eligibleNextSentenceId"] is None
    assert audit["nextSentenceIncluded"] is False


def test_strict_mapping_rejects_text_changes():
    sentence_index, sections = source_data()
    sentence_index[("book", 2)]["text"] = "文本变化。"
    try:
        _materialize_record(
            record(),
            sentence_index,
            sections,
            exact_index(sentence_index),
            source_rows(),
            {},
            CharacterTokenizer(),
            max_length=512,
        )
    except ValueError as error:
        assert "文本或句型不一致" in str(error)
    else:
        raise AssertionError("strict mapping must reject changed text")
