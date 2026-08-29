"""Prepare a bounded second-adjudication pool from large Qwen/agent disagreements."""

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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_unique(rows: list[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("id", ""))
        if not row_id or row_id in result:
            raise ValueError(f"{source}: missing or duplicate id={row_id}")
        result[row_id] = row
    return result


def emotions(row: dict[str, Any]) -> dict[str, float]:
    value = row.get("emotions")
    if not isinstance(value, dict) or set(value) != set(EMOTIONS):
        raise ValueError(f"invalid emotion keys for id={row.get('id')}")
    result = {name: float(value[name]) for name in EMOTIONS}
    if any(number not in LEVELS for number in result.values()):
        raise ValueError(f"invalid emotion levels for id={row.get('id')}")
    return result


def primary(vector: dict[str, float]) -> str:
    active = {name: value for name, value in vector.items() if name != "calm"}
    if not any(active.values()):
        return "base"
    ranked = sorted(active.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) > 1 and ranked[1][1] > 0 and ranked[0][1] - ranked[1][1] <= 0.3300001:
        return "mixed"
    return ranked[0][0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--merged-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=50)
    args = parser.parse_args()
    if args.limit <= 0 or args.chunk_size <= 0 or args.offset < 0:
        raise ValueError("limit 和 chunk-size 必须大于 0，offset 不能为负")

    qwen_rows = index_unique(read_jsonl(args.qwen), "qwen")
    final_paths = [args.final_root / f"{name}.jsonl" for name in ("train", "dev", "test")]
    final_rows = [row for path in final_paths for row in read_jsonl(path)]
    final_by_id = index_unique(final_rows, "final")

    merged_by_id: dict[str, dict[str, Any]] = {}
    for chunk in sorted(args.merged_root.glob("chunk-*")):
        for path in (
            chunk / "accepted-pending-license.jsonl",
            chunk / "reviewed" / "reviewed-accepted.jsonl",
        ):
            if path.is_file():
                merged_by_id.update(index_unique(read_jsonl(path), str(path)))

    scored: list[tuple[float, dict[str, Any]]] = []
    for row_id, gold in final_by_id.items():
        if row_id not in qwen_rows or row_id not in merged_by_id:
            raise ValueError(f"missing comparison row id={row_id}")
        agent = emotions(gold)
        qwen_raw = emotions(qwen_rows[row_id])
        qwen_semantic = dict(qwen_raw)
        qwen_semantic["calm"] = 0.0
        differences = [
            name for name in EMOTIONS if agent[name] != qwen_semantic[name]
        ]
        max_difference = max(abs(agent[name] - qwen_semantic[name]) for name in EMOTIONS)
        if max_difference < 0.67:
            continue
        agent_base = not any(agent.values())
        qwen_base = not any(qwen_semantic.values())
        agent_primary = str(merged_by_id[row_id].get("primaryEmotion", primary(agent)))
        qwen_primary = str(qwen_rows[row_id].get("primaryEmotion", primary(qwen_raw)))
        score = (
            max_difference * 100.0
            + (30.0 if agent_base != qwen_base else 0.0)
            + max(float(gold.get("intensity", 0.0)), float(qwen_rows[row_id].get("intensity", 0.0))) * 10.0
            + (5.0 if agent_primary != qwen_primary else 0.0)
        )
        scored.append(
            (
                score,
                {
                    "schema": "readest-emotion-adjudication-input-v1",
                    "id": row_id,
                    "workId": gold.get("workId"),
                    "sentenceType": gold.get("sentenceType"),
                    "previousText": gold.get("previousText", ""),
                    "text": gold.get("text", ""),
                    "severity": {
                        "score": round(score, 4),
                        "maxAbsoluteDifference": max_difference,
                        "differentDimensions": differences,
                        "baseConflict": agent_base != qwen_base,
                        "primaryConflict": agent_primary != qwen_primary,
                    },
                    "agentJudgment": {
                        "emotions": agent,
                        "intensity": float(gold["intensity"]),
                        "primaryEmotion": agent_primary,
                        "annotationStatus": gold.get("annotationStatus"),
                    },
                    "qwenJudgment": {
                        "emotionsRaw": qwen_raw,
                        "emotionsNeutralAdjusted": qwen_semantic,
                        "intensity": float(qwen_rows[row_id]["intensity"]),
                        "primaryEmotion": qwen_primary,
                        "status": qwen_rows[row_id].get("status"),
                        "confidence": qwen_rows[row_id].get("confidence"),
                        "hardCases": qwen_rows[row_id].get("hardCases", []),
                    },
                    "sourceHashes": {
                        "candidateTextSha256": qwen_rows[row_id].get("textSha256"),
                        "candidatePreviousTextSha256": qwen_rows[row_id].get("previousTextSha256"),
                    },
                },
            )
        )

    scored.sort(key=lambda item: (-item[0], str(item[1]["id"])))
    selected = [row for _, row in scored[args.offset : args.offset + args.limit]]
    output_dir = args.output_dir
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for old in chunks_dir.glob("chunk-*.jsonl"):
        old.unlink()
    for index in range(0, len(selected), args.chunk_size):
        chunk_number = index // args.chunk_size + 1
        path = chunks_dir / f"chunk-{chunk_number:04d}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in selected[index : index + args.chunk_size]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "schema": "readest-emotion-adjudication-pool-v1",
        "selectionRule": "neutral-adjusted max absolute dimension difference >= 0.67",
        "availableSevereCount": len(scored),
        "offset": args.offset,
        "selectedCount": len(selected),
        "chunkSize": args.chunk_size,
        "chunkCount": (len(selected) + args.chunk_size - 1) // args.chunk_size,
        "candidateCount": len(qwen_rows),
        "qwenSha256": sha256(args.qwen),
        "finalSha256": {path.stem: sha256(path) for path in final_paths},
        "works": dict(Counter(str(row.get("workId")) for row in selected)),
        "sentenceTypes": dict(Counter(str(row.get("sentenceType")) for row in selected)),
        "annotationStatuses": dict(Counter(str(row["agentJudgment"].get("annotationStatus")) for row in selected)),
        "baseConflicts": sum(1 for row in selected if row["severity"]["baseConflict"]),
        "primaryConflicts": sum(1 for row in selected if row["severity"]["primaryConflict"]),
    }
    (output_dir / "pool-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
