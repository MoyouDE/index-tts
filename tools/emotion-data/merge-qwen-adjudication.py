"""Validate and merge independent second-adjudication decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


EMOTIONS = (
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
)
LEVELS = {0.0, 0.33, 0.67, 1.0}
PRIMARY = set(EMOTIONS) | {"base", "mixed"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_decision(row: dict[str, Any], source: str) -> None:
    if row.get("schema") != "readest-emotion-adjudication-v1":
        raise ValueError(f"{source}: schema mismatch")
    if not str(row.get("id", "")):
        raise ValueError(f"{source}: missing id")
    if row.get("disposition") not in {"keep", "drop"}:
        raise ValueError(f"{source}: invalid disposition")
    if row.get("chosenSource") not in {"agent", "qwen", "revised"}:
        raise ValueError(f"{source}: invalid chosenSource")
    if row.get("confidence") not in {"high", "medium", "low"}:
        raise ValueError(f"{source}: invalid confidence")
    if not isinstance(row.get("rationale"), str) or not row["rationale"].strip():
        raise ValueError(f"{source}: missing rationale")
    values = row.get("emotions")
    if not isinstance(values, dict) or set(values) != set(EMOTIONS):
        raise ValueError(f"{source}: invalid emotion keys")
    floats = {name: float(values[name]) for name in EMOTIONS}
    if any(value not in LEVELS for value in floats.values()):
        raise ValueError(f"{source}: invalid emotion level")
    intensity = float(row.get("intensity"))
    if intensity not in LEVELS:
        raise ValueError(f"{source}: invalid intensity")
    primary = row.get("primaryEmotion")
    if primary not in PRIMARY:
        raise ValueError(f"{source}: invalid primaryEmotion")
    if not any(floats.values()) and (intensity != 0.0 or primary != "base"):
        raise ValueError(f"{source}: base vector must have zero intensity and base primary")
    if any(floats.values()) and (intensity == 0.0 or primary == "base"):
        raise ValueError(f"{source}: active vector requires non-zero intensity and active primary")
    for field in ("textSha256", "previousTextSha256"):
        value = row.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{source}: invalid {field}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--decisions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_paths = sorted((args.pool_dir / "chunks").glob("chunk-*.jsonl"))
    if not input_paths:
        raise ValueError("no adjudication input chunks")
    merged: list[dict[str, Any]] = []
    chunk_reports: list[dict[str, Any]] = []
    disposition_counts = Counter()
    source_counts = Counter()
    confidence_counts = Counter()
    decision_agent_exact = 0
    decision_qwen_exact = 0
    base_conflict_choice = Counter()
    for input_path in input_paths:
        decision_path = args.decisions_dir / input_path.name
        if not decision_path.is_file():
            raise ValueError(f"missing decision file: {decision_path}")
        inputs = read_jsonl(input_path)
        decisions = read_jsonl(decision_path)
        if len(inputs) != len(decisions):
            raise ValueError(f"{input_path.name}: input/decision row count mismatch")
        input_hash = sha256(input_path)
        decision_hash = sha256(decision_path)
        for index, (input_row, decision) in enumerate(zip(inputs, decisions), 1):
            validate_decision(decision, f"{decision_path}:{index}")
            if decision["id"] != input_row["id"]:
                raise ValueError(f"{decision_path}:{index}: id/order mismatch")
            for field in ("textSha256", "previousTextSha256"):
                if decision[field] != input_row["sourceHashes"].get(
                    "candidate" + field[0].upper() + field[1:]
                ):
                    raise ValueError(f"{decision_path}:{index}: {field} mismatch")
            agent_emotions = input_row["agentJudgment"].get("emotions")
            qwen_emotions = input_row["qwenJudgment"].get("emotionsNeutralAdjusted")
            if not isinstance(agent_emotions, dict) or not isinstance(qwen_emotions, dict):
                raise ValueError(f"{decision_path}:{index}: missing source judgments")
            decision_emotions = decision["emotions"]
            selected_emotions = None
            if decision["chosenSource"] == "agent":
                selected_emotions = agent_emotions
            elif decision["chosenSource"] == "qwen":
                selected_emotions = qwen_emotions
            if selected_emotions is not None and any(
                float(decision_emotions[name]) != float(selected_emotions[name]) for name in EMOTIONS
            ):
                raise ValueError(
                    f"{decision_path}:{index}: chosenSource does not match decision vector"
                )
            decision_agent_exact += int(
                all(float(decision_emotions[name]) == float(agent_emotions[name]) for name in EMOTIONS)
            )
            decision_qwen_exact += int(
                all(float(decision_emotions[name]) == float(qwen_emotions[name]) for name in EMOTIONS)
            )
            agent_base = not any(float(agent_emotions[name]) for name in EMOTIONS)
            qwen_base = not any(float(qwen_emotions[name]) for name in EMOTIONS)
            decision_base = not any(float(decision_emotions[name]) for name in EMOTIONS)
            if agent_base != qwen_base:
                if decision_base == agent_base:
                    base_conflict_choice["agent"] += 1
                elif decision_base == qwen_base:
                    base_conflict_choice["qwen"] += 1
                else:
                    base_conflict_choice["revised"] += 1
            merged.append(
                {
                    "schema": "readest-emotion-adjudication-result-v1",
                    "id": input_row["id"],
                    "workId": input_row.get("workId"),
                    "sentenceType": input_row.get("sentenceType"),
                    "previousText": input_row.get("previousText", ""),
                    "text": input_row.get("text", ""),
                    "agentJudgment": input_row.get("agentJudgment"),
                    "qwenJudgment": input_row.get("qwenJudgment"),
                    "severity": input_row.get("severity"),
                    "secondDecision": decision,
                }
            )
            disposition_counts[str(decision["disposition"])] += 1
            source_counts[str(decision["chosenSource"])] += 1
            confidence_counts[str(decision["confidence"])] += 1
        chunk_reports.append(
            {
                "chunk": input_path.name,
                "rows": len(inputs),
                "inputSha256": input_hash,
                "decisionSha256": decision_hash,
            }
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "adjudicated.jsonl"
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema": "readest-emotion-adjudication-result-v1",
        "inputChunkCount": len(input_paths),
        "inputRowCount": len(merged),
        "decisionRowCount": len(merged),
        "missingChunks": 0,
        "disposition": dict(disposition_counts),
        "chosenSource": dict(source_counts),
        "confidence": dict(confidence_counts),
        "secondDecisionMatchesAgentCount": decision_agent_exact,
        "secondDecisionMatchesAgentRate": decision_agent_exact / len(merged) if merged else 0.0,
        "secondDecisionMatchesQwenNeutralAdjustedCount": decision_qwen_exact,
        "secondDecisionMatchesQwenNeutralAdjustedRate": decision_qwen_exact / len(merged) if merged else 0.0,
        "baseConflictSecondDecisionChoice": dict(base_conflict_choice),
        "chunks": chunk_reports,
        "outputSha256": sha256(output_path),
    }
    (output_dir / "adjudication-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
