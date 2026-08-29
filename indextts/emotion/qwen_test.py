"""Test-only batch pre-annotation with the local IndexTTS Qwen emotion model.

This is deliberately separate from the MacBERT training path.  Its output is
marked ``test-only`` and must be reviewed before it can be used as labels.
The runner is resumable and writes one JSON object per candidate row.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Iterable

from .annotation import ANNOTATION_SCHEMA, file_sha256
from .schema import EMOTION_NAMES
from .source_extract import iter_candidate_records, sha256_text


_CN_NAMES = {
    "高兴": "happy",
    "喜": "happy",
    "愤怒": "angry",
    "怒": "angry",
    "悲伤": "sad",
    "哀": "sad",
    "恐惧": "afraid",
    "惧": "afraid",
    "反感": "disgusted",
    "厌恶": "disgusted",
    "低落": "melancholic",
    "惊讶": "surprised",
    "惊喜": "surprised",
    "自然": "calm",
    "平静": "calm",
}
_LEVELS = (0.0, 0.33, 0.67, 1.0)
_HARD_CASES = {
    "negation": re.compile(r"不|没|无|未|别|莫|难道|却|但|然而|反而|只是|偏偏|哪知|怎料|没想到|原来"),
    "sarcasm": re.compile(r"呵|哼|冷笑|讥|嘲|讽|笑死|可笑|真是"),
    "suspense": re.compile(r"忽然|突然|神秘|诡异|黑暗|背后|危机|危险|杀机|不知|谜|悬"),
    "fright": re.compile(r"恐|害怕|颤|发抖|惊叫|尖叫|逃|心惊|骇"),
    "inner_monologue": re.compile(r"心想|暗道|心中|内心|想着|念头|他想|她想"),
}


def _quantize(value: object) -> float:
    try:
        number = max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
    return min(_LEVELS, key=lambda level: abs(level - number))


def _extract_json(content: str) -> dict[str, object] | None:
    content = content.strip()
    try:
        value = json.loads(content)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", content, flags=re.S)
    if match:
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _emotion_values(payload: dict[str, object]) -> dict[str, float]:
    values = {name: 0.0 for name in EMOTION_NAMES}
    for key, value in payload.items():
        name = _CN_NAMES.get(str(key), str(key))
        if name in values:
            values[name] = _quantize(value)
    return values


def _make_annotation(candidate: dict[str, object], payload: dict[str, object] | None, *, error: str | None = None) -> dict[str, object]:
    values = _emotion_values(payload or {})
    text = f"{candidate.get('previousText', '')} {candidate.get('text', '')}"
    hard_cases = [name for name, pattern in _HARD_CASES.items() if pattern.search(text)]
    invalid = payload is None
    if all(value == 0.0 for value in values.values()):
        primary = "base"
        intensity = 0.0
    else:
        ranked = sorted(values.items(), key=lambda item: item[1], reverse=True)
        intensity = max(values.values())
        primary = ranked[0][0]
        if len(ranked) > 1 and ranked[1][1] > 0.0 and ranked[0][1] - ranked[1][1] <= 0.33:
            primary = "mixed"
    reason = error or "test-only Qwen 批量预标注；需人工复核"
    return {
        "id": candidate["id"],
        "textSha256": candidate["textSha256"],
        "previousTextSha256": candidate["previousTextSha256"],
        "status": "review" if invalid or hard_cases else "accepted",
        "emotions": values,
        "intensity": intensity,
        "primaryEmotion": primary,
        "confidence": "low" if invalid or hard_cases else "medium",
        "needsReview": True,
        "hardCases": hard_cases,
        "rationale": reason,
    }


def annotate_qwen_test(
    candidates_path: str | Path,
    output_path: str | Path,
    model_dir: str | Path,
    *,
    batch_size: int = 8,
    max_new_tokens: int = 128,
    prompt_variant: str = "a",
) -> dict[str, object]:
    """Annotate all candidates with local Qwen, resuming a partial JSONL file."""
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if prompt_variant not in {"a", "b"}:
        raise ValueError("prompt_variant 只能是 a 或 b")
    candidates = list(iter_candidate_records(candidates_path))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict[str, object]] = {}
    if output.is_file():
        with output.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, dict) and row.get("id"):
                        done[str(row["id"])] = row
    pending = [candidate for candidate in candidates if str(candidate["id"]) not in done]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    mode_hint = (
        "重点结合上一句，谨慎处理否定、转折和反讽。"
        if prompt_variant == "a"
        else "优先判断当前句主动情绪；普通中性句必须全部为0，不要把calm当中性。"
    )
    with output.open("a", encoding="utf-8", newline="\n") as handle:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            prompts = []
            for candidate in batch:
                prompts.append(
                    tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": "文本情感分类"},
                            {
                                "role": "user",
                                "content": (
                                    "请只输出JSON对象，键为高兴、愤怒、悲伤、恐惧、反感、低落、惊讶、自然，"
                                    "值为0到1之间数字。\n"
                                    f"{mode_hint}\n上一句：{candidate.get('previousText', '')}\n"
                                    f"当前句（{candidate.get('sentenceType', '')}）：{candidate.get('text', '')}"
                                ),
                            },
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            prompt_length = encoded.input_ids.shape[1]
            decoded = tokenizer.batch_decode(
                generated[:, prompt_length:], skip_special_tokens=True
            )
            for candidate, content in zip(batch, decoded):
                annotation = _make_annotation(candidate, _extract_json(content))
                handle.write(json.dumps(annotation, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

    report = {
        "schema": ANNOTATION_SCHEMA,
        "mode": "test-only-qwen-pseudolabel",
        "promptVariant": prompt_variant,
        "candidateFileSha256": file_sha256(candidates_path),
        "annotationFileSha256": file_sha256(output),
        "candidateCount": len(candidates),
        "annotatedCount": len(candidates),
        "modelDir": str(Path(model_dir).resolve()),
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
