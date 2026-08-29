"""Compare Qwen emotion predictions with the finalized agent-reviewed labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_emotions(row: dict[str, Any], source: str) -> dict[str, float]:
    emotions = row.get("emotions")
    if not isinstance(emotions, dict) or set(emotions) != set(EMOTIONS):
        raise ValueError(f"{source}: invalid emotion keys for id={row.get('id')}")
    result = {name: float(emotions[name]) for name in EMOTIONS}
    if any(value not in LEVELS for value in result.values()):
        raise ValueError(f"{source}: invalid emotion level for id={row.get('id')}")
    return result


def index_unique(rows: Iterable[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("id", ""))
        if not row_id:
            raise ValueError(f"{source}: row without id")
        if row_id in result:
            raise ValueError(f"{source}: duplicate id={row_id}")
        result[row_id] = row
    return result


def load_gold_primary(merged_root: Path) -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk in sorted(merged_root.glob("chunk-*")):
        direct = chunk / "accepted-pending-license.jsonl"
        reviewed = chunk / "reviewed" / "reviewed-accepted.jsonl"
        if direct.is_file():
            rows.extend(read_jsonl(direct))
        if reviewed.is_file():
            rows.extend(read_jsonl(reviewed))
    return index_unique(rows, "merged gold labels")


def ratio(value: int, total: int) -> float:
    return value / total if total else 0.0


def inferred_primary(emotions: dict[str, float]) -> str:
    if not any(emotions.values()):
        return "base"
    ranked = sorted(emotions.items(), key=lambda item: item[1], reverse=True)
    if (
        len(ranked) > 1
        and ranked[1][1] > 0.0
        and ranked[0][1] - ranked[1][1] <= 0.3300001
    ):
        return "mixed"
    return ranked[0][0]


def compare_reference(
    reference_by_id: dict[str, dict[str, Any]],
    qwen_by_id: dict[str, dict[str, Any]],
    *,
    neutral_adjusted: bool,
) -> dict[str, Any]:
    exact_count = 0
    primary_match_count = 0
    base_conflicts = Counter()
    by_status: dict[str, Counter[str]] = defaultdict(Counter)
    for row_id, reference in reference_by_id.items():
        qwen = qwen_by_id[row_id]
        reference_emotions = as_emotions(reference, "agent pass")
        qwen_emotions = as_emotions(qwen, "qwen")
        if neutral_adjusted:
            qwen_emotions["calm"] = 0.0
        exact = reference_emotions == qwen_emotions
        exact_count += int(exact)
        reference_primary = str(reference.get("primaryEmotion", ""))
        qwen_primary = (
            inferred_primary(qwen_emotions)
            if neutral_adjusted
            else str(qwen.get("primaryEmotion", ""))
        )
        primary_match = reference_primary == qwen_primary
        primary_match_count += int(primary_match)
        reference_base = not any(reference_emotions.values())
        qwen_base = not any(qwen_emotions.values())
        if reference_base and not qwen_base:
            base_conflicts["agentBaseQwenActive"] += 1
        elif not reference_base and qwen_base:
            base_conflicts["agentActiveQwenBase"] += 1
        elif reference_base:
            base_conflicts["bothBase"] += 1
        else:
            base_conflicts["bothActive"] += 1
        status = str(reference.get("status", "unknown"))
        by_status[status]["count"] += 1
        by_status[status]["exact"] += int(exact)
        by_status[status]["primaryMatch"] += int(primary_match)

    total = len(reference_by_id)
    conflict_count = base_conflicts["agentBaseQwenActive"] + base_conflicts["agentActiveQwenBase"]
    return {
        "count": total,
        "exactVectorMatchCount": exact_count,
        "exactVectorAgreementRate": ratio(exact_count, total),
        "differentVectorCount": total - exact_count,
        "differentVectorRate": ratio(total - exact_count, total),
        "primaryMatchCount": primary_match_count,
        "primaryAgreementRate": ratio(primary_match_count, total),
        "primaryDifferentCount": total - primary_match_count,
        "baseActiveConflictCount": conflict_count,
        "baseActiveConflictRate": ratio(conflict_count, total),
        "baseBreakdown": dict(base_conflicts),
        "byAgentStatus": {
            status: {
                "count": values["count"],
                "exactVectorMatches": values["exact"],
                "exactVectorAgreementRate": ratio(values["exact"], values["count"]),
                "primaryMatches": values["primaryMatch"],
                "primaryAgreementRate": ratio(values["primaryMatch"], values["count"]),
            }
            for status, values in sorted(by_status.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--merged-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotation-a-dir", type=Path)
    parser.add_argument("--annotation-b-dir", type=Path)
    args = parser.parse_args()

    candidates = read_jsonl(args.candidates)
    qwen_rows = read_jsonl(args.qwen)
    candidate_by_id = index_unique(candidates, "candidates")
    qwen_by_id = index_unique(qwen_rows, "qwen")
    if len(candidates) != len(qwen_rows) or set(candidate_by_id) != set(qwen_by_id):
        raise ValueError("Qwen output does not cover the complete candidate set")
    for row_id, qwen in qwen_by_id.items():
        candidate = candidate_by_id[row_id]
        for field in ("textSha256", "previousTextSha256"):
            if qwen.get(field) != candidate.get(field):
                raise ValueError(f"Qwen {field} mismatch for id={row_id}")
        as_emotions(qwen, "qwen")

    final_paths = [args.final_root / f"{split}.jsonl" for split in ("train", "dev", "test")]
    final_rows = [row for path in final_paths for row in read_jsonl(path)]
    final_by_id = index_unique(final_rows, "final labels")
    gold_source_by_id = load_gold_primary(args.merged_root)
    if not set(final_by_id).issubset(qwen_by_id):
        raise ValueError("Final labels contain ids missing from Qwen output")
    if set(final_by_id) != set(gold_source_by_id):
        raise ValueError("Final labels and merged accepted labels do not have the same ids")

    all_candidate_agent_pass_comparison: dict[str, Any] = {}
    for pass_name, annotation_dir in (
        ("passA", args.annotation_a_dir),
        ("passB", args.annotation_b_dir),
    ):
        if annotation_dir is None:
            continue
        pass_rows = [
            row
            for path in sorted(annotation_dir.glob("chunk-*.jsonl"))
            for row in read_jsonl(path)
        ]
        pass_by_id = index_unique(pass_rows, pass_name)
        if set(pass_by_id) != set(qwen_by_id):
            raise ValueError(f"{pass_name} does not cover the complete Qwen candidate set")
        all_candidate_agent_pass_comparison[pass_name] = {
            "rawTtsMapping": compare_reference(
                pass_by_id, qwen_by_id, neutral_adjusted=False
            ),
            "neutralAdjusted": compare_reference(
                pass_by_id, qwen_by_id, neutral_adjusted=True
            ),
        }

    exact_count = 0
    primary_match_count = 0
    base_conflicts = Counter()
    intensity_difference_count = 0
    dimension_differences = Counter({name: 0 for name in EMOTIONS})
    dimension_absolute_error = Counter({name: 0.0 for name in EMOTIONS})
    different_dimension_cardinality = Counter()
    qwen_status = Counter()
    by_status: dict[str, Counter[str]] = defaultdict(Counter)
    by_sentence_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_work: dict[str, Counter[str]] = defaultdict(Counter)
    disagreements: list[dict[str, Any]] = []
    semantic_exact_count = 0
    semantic_primary_match_count = 0
    semantic_intensity_difference_count = 0
    semantic_base_conflicts = Counter()
    semantic_dimension_differences = Counter({name: 0 for name in EMOTIONS})
    semantic_different_dimension_cardinality = Counter()
    semantic_disagreements: list[dict[str, Any]] = []

    for row_id, gold in final_by_id.items():
        qwen = qwen_by_id[row_id]
        source_gold = gold_source_by_id[row_id]
        gold_emotions = as_emotions(gold, "final labels")
        qwen_emotions = as_emotions(qwen, "qwen")
        semantic_qwen_emotions = dict(qwen_emotions)
        # IndexTTS Qwen calls its neutral/default output "natural" and maps it
        # to the TTS calm basis.  The classifier dataset reserves calm for an
        # explicitly soothing/composed emotion, so neutral-aware comparison
        # must remove this runtime-only natural component.
        semantic_qwen_emotions["calm"] = 0.0
        if gold_emotions != as_emotions(source_gold, "merged gold labels"):
            raise ValueError(f"Final and merged emotion vectors differ for id={row_id}")

        different = [
            name for name in EMOTIONS if gold_emotions[name] != qwen_emotions[name]
        ]
        exact = not different
        exact_count += int(exact)
        different_dimension_cardinality[len(different)] += 1
        for name in EMOTIONS:
            delta = abs(gold_emotions[name] - qwen_emotions[name])
            dimension_absolute_error[name] += delta
            dimension_differences[name] += int(delta > 0.0)

        gold_primary = str(source_gold.get("primaryEmotion", ""))
        qwen_primary = str(qwen.get("primaryEmotion", ""))
        primary_match = gold_primary == qwen_primary
        primary_match_count += int(primary_match)
        gold_base = all(value == 0.0 for value in gold_emotions.values())
        qwen_base = all(value == 0.0 for value in qwen_emotions.values())
        if gold_base and not qwen_base:
            base_conflicts["goldBaseQwenActive"] += 1
        elif not gold_base and qwen_base:
            base_conflicts["goldActiveQwenBase"] += 1
        elif gold_base:
            base_conflicts["bothBase"] += 1
        else:
            base_conflicts["bothActive"] += 1

        intensity_different = float(gold["intensity"]) != float(qwen["intensity"])
        intensity_difference_count += int(intensity_different)
        qwen_status[str(qwen.get("status", ""))] += 1

        semantic_different = [
            name
            for name in EMOTIONS
            if gold_emotions[name] != semantic_qwen_emotions[name]
        ]
        semantic_exact = not semantic_different
        semantic_exact_count += int(semantic_exact)
        semantic_different_dimension_cardinality[len(semantic_different)] += 1
        for name in semantic_different:
            semantic_dimension_differences[name] += 1
        semantic_primary = inferred_primary(semantic_qwen_emotions)
        semantic_primary_match = gold_primary == semantic_primary
        semantic_primary_match_count += int(semantic_primary_match)
        semantic_qwen_base = all(value == 0.0 for value in semantic_qwen_emotions.values())
        if gold_base and not semantic_qwen_base:
            semantic_base_conflicts["goldBaseQwenActive"] += 1
        elif not gold_base and semantic_qwen_base:
            semantic_base_conflicts["goldActiveQwenBase"] += 1
        elif gold_base:
            semantic_base_conflicts["bothBase"] += 1
        else:
            semantic_base_conflicts["bothActive"] += 1
        semantic_qwen_intensity = max(semantic_qwen_emotions.values())
        semantic_intensity_difference_count += int(
            float(gold["intensity"]) != semantic_qwen_intensity
        )

        groups = (
            (by_status[str(gold.get("annotationStatus", "unknown"))], exact, primary_match),
            (by_sentence_type[str(gold.get("sentenceType", "unknown"))], exact, primary_match),
            (by_work[str(gold.get("workId", "unknown"))], exact, primary_match),
        )
        for counter, is_exact, is_primary_match in groups:
            counter["count"] += 1
            counter["exact"] += int(is_exact)
            counter["primaryMatch"] += int(is_primary_match)

        if not exact:
            disagreements.append(
                {
                    "id": row_id,
                    "workId": gold.get("workId"),
                    "sentenceType": gold.get("sentenceType"),
                    "annotationStatus": gold.get("annotationStatus"),
                    "previousText": gold.get("previousText"),
                    "text": gold.get("text"),
                    "differentDimensions": different,
                    "l1Difference": round(
                        sum(abs(gold_emotions[name] - qwen_emotions[name]) for name in EMOTIONS),
                        4,
                    ),
                    "maxAbsoluteDifference": max(
                        abs(gold_emotions[name] - qwen_emotions[name]) for name in EMOTIONS
                    ),
                    "primaryDifferent": not primary_match,
                    "baseActiveConflict": gold_base != qwen_base,
                    "gold": {
                        "emotions": gold_emotions,
                        "intensity": float(gold["intensity"]),
                        "primaryEmotion": gold_primary,
                    },
                    "qwen": {
                        "emotions": qwen_emotions,
                        "intensity": float(qwen["intensity"]),
                        "primaryEmotion": qwen_primary,
                        "status": qwen.get("status"),
                        "confidence": qwen.get("confidence"),
                        "hardCases": qwen.get("hardCases", []),
                    },
                }
            )
        if not semantic_exact:
            semantic_disagreements.append(
                {
                    "id": row_id,
                    "workId": gold.get("workId"),
                    "sentenceType": gold.get("sentenceType"),
                    "annotationStatus": gold.get("annotationStatus"),
                    "previousText": gold.get("previousText"),
                    "text": gold.get("text"),
                    "differentDimensions": semantic_different,
                    "primaryDifferent": not semantic_primary_match,
                    "baseActiveConflict": gold_base != semantic_qwen_base,
                    "gold": {
                        "emotions": gold_emotions,
                        "intensity": float(gold["intensity"]),
                        "primaryEmotion": gold_primary,
                    },
                    "qwenNeutralAdjusted": {
                        "emotions": semantic_qwen_emotions,
                        "intensity": semantic_qwen_intensity,
                        "primaryEmotion": semantic_primary,
                    },
                }
            )

    total = len(final_rows)

    def group_report(groups: dict[str, Counter[str]]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for name, values in sorted(groups.items()):
            count = values["count"]
            exact = values["exact"]
            primary = values["primaryMatch"]
            result[name] = {
                "count": count,
                "exactVectorMatches": exact,
                "exactVectorAgreementRate": ratio(exact, count),
                "primaryMatches": primary,
                "primaryAgreementRate": ratio(primary, count),
            }
        return result

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    disagreements_path = output_dir / "disagreements.jsonl"
    with disagreements_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in disagreements:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    semantic_disagreements_path = output_dir / "neutral-adjusted-disagreements.jsonl"
    with semantic_disagreements_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in semantic_disagreements:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    primary_difference_count = total - primary_match_count
    exact_difference_count = total - exact_count
    base_active_conflict_count = (
        base_conflicts["goldBaseQwenActive"] + base_conflicts["goldActiveQwenBase"]
    )
    semantic_difference_count = total - semantic_exact_count
    semantic_primary_difference_count = total - semantic_primary_match_count
    semantic_base_active_conflict_count = (
        semantic_base_conflicts["goldBaseQwenActive"]
        + semantic_base_conflicts["goldActiveQwenBase"]
    )
    report = {
        "schema": "readest-emotion-qwen-comparison-v1",
        "candidateCount": len(candidates),
        "qwenCount": len(qwen_rows),
        "comparedFinalCount": total,
        "exactVectorMatchCount": exact_count,
        "exactVectorAgreementRate": ratio(exact_count, total),
        "differentVectorCount": exact_difference_count,
        "differentVectorRate": ratio(exact_difference_count, total),
        "primaryMatchCount": primary_match_count,
        "primaryAgreementRate": ratio(primary_match_count, total),
        "primaryDifferentCount": primary_difference_count,
        "baseActiveConflictCount": base_active_conflict_count,
        "baseActiveConflictRate": ratio(base_active_conflict_count, total),
        "baseBreakdown": dict(base_conflicts),
        "intensityDifferentCount": intensity_difference_count,
        "intensityDifferentRate": ratio(intensity_difference_count, total),
        "dimensionDifferenceCount": dict(dimension_differences),
        "dimensionAgreementRate": {
            name: ratio(total - dimension_differences[name], total) for name in EMOTIONS
        },
        "dimensionMeanAbsoluteError": {
            name: dimension_absolute_error[name] / total for name in EMOTIONS
        },
        "differentDimensionCardinality": {
            str(key): value for key, value in sorted(different_dimension_cardinality.items())
        },
        "qwenStatus": dict(qwen_status),
        "allCandidateAgentPassComparison": all_candidate_agent_pass_comparison,
        "neutralAdjustedComparison": {
            "note": "Qwen 的自然表示中性/base，不作为训练标签中的主动 calm。",
            "exactVectorMatchCount": semantic_exact_count,
            "exactVectorAgreementRate": ratio(semantic_exact_count, total),
            "differentVectorCount": semantic_difference_count,
            "differentVectorRate": ratio(semantic_difference_count, total),
            "primaryMatchCount": semantic_primary_match_count,
            "primaryAgreementRate": ratio(semantic_primary_match_count, total),
            "primaryDifferentCount": semantic_primary_difference_count,
            "baseActiveConflictCount": semantic_base_active_conflict_count,
            "baseActiveConflictRate": ratio(semantic_base_active_conflict_count, total),
            "baseBreakdown": dict(semantic_base_conflicts),
            "intensityDifferentCount": semantic_intensity_difference_count,
            "intensityDifferentRate": ratio(semantic_intensity_difference_count, total),
            "dimensionDifferenceCount": dict(semantic_dimension_differences),
            "dimensionAgreementRate": {
                name: ratio(total - semantic_dimension_differences[name], total)
                for name in EMOTIONS
            },
            "differentDimensionCardinality": {
                str(key): value
                for key, value in sorted(semantic_different_dimension_cardinality.items())
            },
        },
        "byAnnotationStatus": group_report(by_status),
        "bySentenceType": group_report(by_sentence_type),
        "byWork": group_report(by_work),
        "inputs": {
            "candidatesSha256": file_sha256(args.candidates),
            "qwenSha256": file_sha256(args.qwen),
            "finalSha256": {path.stem: file_sha256(path) for path in final_paths},
        },
        "outputs": {
            "disagreements": str(disagreements_path.resolve()),
            "disagreementsSha256": file_sha256(disagreements_path),
            "neutralAdjustedDisagreements": str(semantic_disagreements_path.resolve()),
            "neutralAdjustedDisagreementsSha256": file_sha256(
                semantic_disagreements_path
            ),
        },
    }
    report_path = output_dir / "comparison-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
