"""Recollect source structure and materialize target-marked emotion contexts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .context_policy import (
    CONTEXT_ENCODING,
    CONTEXT_MAX_LENGTH,
    CONTEXT_POLICY,
    EMOTION_ABI,
    EMOTION_SPECIAL_TOKENS,
    ensure_emotion_special_tokens,
    normalize_context_sentence,
    select_target_context,
)
from .continuous_v3_context import (
    EXPECTED_NOVEL_SPLITS,
    _load_novel_records,
    _load_source_rows,
    _record_location,
)
from .schema import EMOTION_NAMES, TARGET_CONTEXT_SCHEMA, TARGET_CONTEXT_SCHEMA_VERSION


AUDIT_SCHEMA = "readest-emotion-target-context-audit-v1"
SOURCE_SPEC_SCHEMA = "readest-emotion-source-collection-spec-v1"
SOURCE_SECTION_SCHEMA = "readest-emotion-source-section-v1"
_ID = re.compile(r"^emotion-candidate-.+-r\d{6}-s(?P<sentence>\d{6})$")
_PSEUDO_WORKS = {"CSI", "JY", "WP2021"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(record: object) -> str:
    value = json.dumps(
        record, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 解析失败 {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL 行必须是对象: {path}:{line_number}")
            yield value


def _write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            count += 1
    return count


def _percentile(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return int(ordered[round((len(ordered) - 1) * fraction)])


def _source_specs(project_root: Path, work_ids: Sequence[str]) -> list[dict[str, object]]:
    speaker_root = project_root / "speaker-id"
    raw_root = speaker_root / "RawNoval"
    data_root = speaker_root / "typescript" / "data"
    stats_root = data_root / "speaker_stats"
    specs: list[dict[str, object]] = []
    for work_id in sorted(work_ids):
        if work_id in _PSEUDO_WORKS:
            ready = data_root / "ExtraData" / "collect_ready" / work_id
            input_path = ready / f"{work_id}_pseudo_original_clean.txt"
            section_manifest = ready / "manifest_clean.json"
            config_candidates = [
                stats_root / f"{work_id}_pseudo_original_clean.json",
                stats_root / f"{work_id}.json",
            ]
        else:
            input_path = raw_root / f"{work_id}.txt"
            section_manifest = None
            config_candidates = [stats_root / f"{work_id}.json"]
        if not input_path.is_file():
            raise ValueError(f"缺少小说原文: {input_path}")
        config_path = next((path for path in config_candidates if path.is_file()), None)
        spec: dict[str, object] = {
            "workId": work_id,
            "inputPath": input_path.resolve().as_posix(),
        }
        if config_path is not None:
            spec["configPath"] = config_path.resolve().as_posix()
        if section_manifest is not None:
            if not section_manifest.is_file():
                raise ValueError(f"缺少 clean corpus manifest: {section_manifest}")
            spec["sectionManifestPath"] = section_manifest.resolve().as_posix()
        specs.append(spec)
    return specs


def collect_source_sections(
    project_root: Path,
    work_ids: Sequence[str],
    output_path: Path,
) -> dict[str, object]:
    """Invoke the speaker TypeScript collector and return its verified manifest."""

    typescript_root = project_root / "speaker-id" / "typescript"
    collector = typescript_root / "src" / "emotionContextCollect.ts"
    if not collector.is_file():
        raise ValueError(f"缺少情感原文采集器: {collector}")
    spec_path = output_path.with_name("source-collection-spec.json")
    spec = {
        "schema": SOURCE_SPEC_SCHEMA,
        "output": output_path.resolve().as_posix(),
        "works": _source_specs(project_root, work_ids),
    }
    spec_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    npm = "npm.cmd" if os.name == "nt" else "npm"
    subprocess.run(
        [
            npm,
            "exec",
            "--",
            "tsx",
            "src/emotionContextCollect.ts",
            "--spec",
            str(spec_path.resolve()),
        ],
        cwd=typescript_root,
        check=True,
    )
    manifest_path = Path(str(output_path) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "readest-emotion-source-sections-manifest-v1"
        or manifest.get("outputSha256") != _sha256_file(output_path)
    ):
        raise ValueError("原文采集 manifest 无效")
    # The collector writes into an atomic temporary tree.  Keep the audit
    # replayable after that tree is renamed by replacing only the two volatile
    # artifact paths with their stable locations inside the final snapshot.
    spec["output"] = "audit/source-sections.jsonl"
    spec_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["specPath"] = "audit/source-collection-spec.json"
    manifest["specSha256"] = _sha256_file(spec_path)
    manifest["output"] = "audit/source-sections.jsonl"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_clean_label_sections(
    project_root: Path,
    work_ids: Sequence[str],
) -> tuple[dict[tuple[str, int], set[str]], dict[str, dict[str, object]]]:
    mapping: dict[tuple[str, int], set[str]] = defaultdict(set)
    sources: dict[str, dict[str, object]] = {}
    ready_root = project_root / "speaker-id" / "typescript" / "data" / "ExtraData" / "collect_ready"
    for work_id in sorted(set(work_ids).intersection(_PSEUDO_WORKS)):
        path = ready_root / work_id / f"{work_id}_labels_clean.jsonl"
        if not path.is_file():
            raise ValueError(f"缺少 clean label manifest: {path}")
        count = 0
        for record in _iter_jsonl(path):
            source = record.get("source")
            chunk = record.get("pseudo_chunk")
            article_index = source.get("article_index") if isinstance(source, Mapping) else None
            if (
                isinstance(article_index, bool)
                or not isinstance(article_index, int)
                or not isinstance(chunk, str)
                or not chunk
            ):
                raise ValueError(f"clean label manifest 行无效: {path}")
            mapping[(work_id, article_index)].add(f"{work_id}:{chunk}")
            count += 1
        sources[work_id] = {
            "path": path.resolve().as_posix(),
            "sha256": _sha256_file(path),
            "rows": count,
        }
    return mapping, sources


def _source_sentence_id(record: Mapping[str, object]) -> int:
    match = _ID.fullmatch(str(record.get("id", "")))
    if match is None:
        raise ValueError(f"无法从记录 ID 提取 sentenceId: {record.get('id')}")
    return int(match.group("sentence"))


def _load_sections(
    path: Path,
) -> tuple[
    dict[tuple[str, int], dict[str, object]],
    dict[str, list[dict[str, object]]],
    dict[tuple[str, str, str], list[dict[str, object]]],
]:
    sentence_index: dict[tuple[str, int], dict[str, object]] = {}
    sections: dict[str, list[dict[str, object]]] = {}
    exact_index: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for record in _iter_jsonl(path):
        if record.get("schema") != SOURCE_SECTION_SCHEMA:
            raise ValueError("原文 section schema 无效")
        work_id = str(record.get("workId", ""))
        section_id = str(record.get("sectionId", ""))
        values = record.get("sentences")
        if not work_id or not section_id or not isinstance(values, list) or not values:
            raise ValueError(f"原文 section 结构无效: {section_id!r}")
        if section_id in sections:
            raise ValueError(f"原文 sectionId 重复: {section_id}")
        section: list[dict[str, object]] = []
        for value in values:
            if not isinstance(value, dict):
                raise ValueError(f"原文 sentence 不是对象: {section_id}")
            sentence_id = value.get("sentenceId")
            if isinstance(sentence_id, bool) or not isinstance(sentence_id, int):
                raise ValueError(f"原文 sentenceId 无效: {section_id}")
            if value.get("sectionId") != section_id:
                raise ValueError(f"原文 sentence sectionId 不一致: {section_id}")
            key = (work_id, sentence_id)
            if key in sentence_index:
                raise ValueError(f"原文 sentenceId 重复: {key}")
            sentence = {
                "sentenceId": sentence_id,
                "sectionId": section_id,
                "chapterIndex": value.get("chapterIndex"),
                "lineIndex": value.get("lineIndex"),
                "text": value.get("text"),
                "sentenceType": value.get("sentenceType"),
                "sourceOffset": value.get("sourceOffset"),
            }
            sentence_index[key] = sentence
            exact_index[
                (
                    work_id,
                    normalize_context_sentence(sentence["text"]),
                    str(sentence["sentenceType"]),
                )
            ].append(sentence)
            section.append(sentence)
        sections[section_id] = section
    return sentence_index, sections, exact_index


def _count_tokens(tokenizer, text: str) -> int:
    return len(
        tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            verbose=False,
        )["input_ids"]
    )


def _same_text_and_type(
    source: Mapping[str, object], reference: Mapping[str, object]
) -> bool:
    return (
        normalize_context_sentence(source.get("text"))
        == normalize_context_sentence(reference.get("text"))
        and source.get("sentenceType") == reference.get("type")
    )


def _resolve_unique_exact_neighbor_alignment(
    candidates: Sequence[Mapping[str, object]],
    sections: Mapping[str, Sequence[Mapping[str, object]]],
    old_sentences: Sequence[object],
    old_target: Mapping[str, object],
) -> Mapping[str, object] | None:
    """Disambiguate duplicate exact text using exact relative neighbors only."""

    old_position = next(
        index for index, sentence in enumerate(old_sentences) if sentence is old_target
    )
    remaining = list(candidates)
    positions = {
        id(sentence): (section, index)
        for section in sections.values()
        for index, sentence in enumerate(section)
    }
    for radius in range(1, len(old_sentences)):
        for direction in (-1, 1):
            old_index = old_position + direction * radius
            if not 0 <= old_index < len(old_sentences):
                continue
            reference = old_sentences[old_index]
            if not isinstance(reference, Mapping):
                continue
            filtered = []
            for candidate in remaining:
                section, candidate_position = positions[id(candidate)]
                raw_index = candidate_position + direction * radius
                if (
                    0 <= raw_index < len(section)
                    and _same_text_and_type(section[raw_index], reference)
                ):
                    filtered.append(candidate)
            if len(filtered) == 1:
                return filtered[0]
            if filtered:
                remaining = filtered
    return None


def _resolve_unique_exact_window_section(
    candidates: Sequence[Mapping[str, object]],
    sections: Mapping[str, Sequence[Mapping[str, object]]],
    old_sentences: Sequence[object],
) -> str | None:
    """Locate one section using multiple exact sentence/type anchors from the old window."""

    old_signatures = {
        (normalize_context_sentence(sentence.get("text")), str(sentence.get("type")))
        for sentence in old_sentences
        if isinstance(sentence, Mapping) and normalize_context_sentence(sentence.get("text"))
    }
    candidate_sections = {str(candidate["sectionId"]) for candidate in candidates}
    evidence: list[tuple[int, str]] = []
    for section_id in candidate_sections:
        signatures = {
            (
                normalize_context_sentence(sentence.get("text")),
                str(sentence.get("sentenceType")),
            )
            for sentence in sections[section_id]
        }
        evidence.append((len(old_signatures.intersection(signatures)), section_id))
    evidence.sort(reverse=True)
    if not evidence or evidence[0][0] < 2:
        return None
    if len(evidence) > 1 and evidence[0][0] == evidence[1][0]:
        return None
    return evidence[0][1]


def _resolve_recollected_sources(
    records_by_split: Mapping[str, Sequence[Mapping[str, object]]],
    sentence_index: Mapping[tuple[str, int], Mapping[str, object]],
    exact_index: Mapping[tuple[str, str, str], Sequence[Mapping[str, object]]],
    sections: Mapping[str, Sequence[Mapping[str, object]]],
    source_rows: Mapping[tuple[str, int], Mapping[str, object]],
    clean_label_sections: Mapping[tuple[str, int], set[str]],
) -> dict[str, tuple[Mapping[str, object], str]]:
    """Map old IDs onto recollected IDs using exact content and monotonic order."""

    by_work: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for records in records_by_split.values():
        for record in records:
            by_work[str(record["workId"])].append(record)
    resolved: dict[str, tuple[Mapping[str, object], str]] = {}
    for work_id, records in by_work.items():
        ordered = sorted(records, key=_source_sentence_id)
        candidates_by_id: dict[str, list[Mapping[str, object]]] = {}
        for record in ordered:
            record_id = str(record["id"])
            old_id = _source_sentence_id(record)
            direct = sentence_index.get((work_id, old_id))
            if (
                direct is not None
                and direct.get("text") == record.get("text")
                and direct.get("sentenceType") == record.get("sentenceType")
                and work_id not in _PSEUDO_WORKS
            ):
                resolved[record_id] = (direct, "global-sentence-id")
                continue
            candidates = list(
                exact_index.get(
                    (
                        work_id,
                        normalize_context_sentence(record.get("text")),
                        str(record.get("sentenceType")),
                    ),
                    (),
                )
            )
            if not candidates and work_id in _PSEUDO_WORKS:
                candidates = _find_exact_concatenated_spans(
                    work_id,
                    normalize_context_sentence(record.get("text")),
                    str(record.get("sentenceType")),
                    sentence_index,
                )
            candidates_by_id[record_id] = candidates
            if len(candidates) == 1 and work_id not in _PSEUDO_WORKS:
                method = (
                    "work-exact-concatenated-span"
                    if candidates[0].get("spanSentenceIds")
                    else "work-exact-text-type"
                )
                resolved[record_id] = (candidates[0], method)

        changed = True
        while changed:
            changed = False
            for index, record in enumerate(ordered):
                record_id = str(record["id"])
                if record_id in resolved:
                    continue
                candidates = candidates_by_id[record_id]
                source_stem, source_row_index, old_target_id = _record_location(record)
                source_row = source_rows.get((source_stem, source_row_index))
                old_sentences = source_row.get("sentences") if source_row else None
                if candidates and isinstance(old_sentences, list):
                    manifest_aligned = _resolve_by_clean_manifest_instance(
                        candidates, work_id, old_sentences, clean_label_sections
                    )
                    if manifest_aligned is not None:
                        resolved[record_id] = (
                            manifest_aligned,
                            "work-exact-text-type-clean-manifest-aligned",
                        )
                        changed = True
                        continue
                    instance_aligned = _resolve_by_instance_header(
                        candidates, sections, old_sentences
                    )
                    if instance_aligned is not None:
                        resolved[record_id] = (
                            instance_aligned,
                            "work-exact-text-type-instance-header-aligned",
                        )
                        changed = True
                        continue
                    previous_aligned = _resolve_by_exact_previous_text(
                        candidates,
                        work_id,
                        normalize_context_sentence(record.get("previousText")),
                        sentence_index,
                    )
                    if previous_aligned is not None:
                        resolved[record_id] = (
                            previous_aligned,
                            "work-exact-text-type-previous-text-aligned",
                        )
                        changed = True
                        continue
                    aligned_section = _resolve_unique_exact_window_section(
                        candidates, sections, old_sentences
                    )
                    if aligned_section is not None:
                        section_candidates = [
                            candidate
                            for candidate in candidates
                            if candidate.get("sectionId") == aligned_section
                        ]
                        if len(section_candidates) == 1:
                            resolved[record_id] = (
                                section_candidates[0],
                                "work-exact-text-type-window-aligned",
                            )
                            changed = True
                            continue
                if len(candidates) == 1:
                    method = (
                        "work-exact-concatenated-span"
                        if candidates[0].get("spanSentenceIds")
                        else "work-exact-text-type"
                    )
                    resolved[record_id] = (candidates[0], method)
                    changed = True
                    continue
                lower: int | None = None
                upper: int | None = None
                for neighbor in reversed(ordered[:index]):
                    value = resolved.get(str(neighbor["id"]))
                    if value is not None:
                        lower = int(value[0]["sentenceId"])
                        break
                for neighbor in ordered[index + 1 :]:
                    value = resolved.get(str(neighbor["id"]))
                    if value is not None:
                        upper = int(value[0]["sentenceId"])
                        break
                bounded = [
                    candidate
                    for candidate in candidates
                    if (lower is None or int(candidate["sentenceId"]) > lower)
                    and (upper is None or int(candidate["sentenceId"]) < upper)
                ]
                if len(bounded) == 1:
                    resolved[record_id] = (bounded[0], "work-exact-text-type-monotonic")
                    changed = True
                    continue
                if len(bounded) > 1:
                    if isinstance(old_sentences, list):
                        manifest_aligned = _resolve_by_clean_manifest_instance(
                            bounded, work_id, old_sentences, clean_label_sections
                        )
                        if manifest_aligned is not None:
                            resolved[record_id] = (
                                manifest_aligned,
                                "work-exact-text-type-clean-manifest-aligned",
                            )
                            changed = True
                            continue
                        instance_aligned = _resolve_by_instance_header(
                            bounded, sections, old_sentences
                        )
                        if instance_aligned is not None:
                            resolved[record_id] = (
                                instance_aligned,
                                "work-exact-text-type-instance-header-aligned",
                            )
                            changed = True
                            continue
                        previous_aligned = _resolve_by_exact_previous_text(
                            bounded,
                            work_id,
                            normalize_context_sentence(record.get("previousText")),
                            sentence_index,
                        )
                        if previous_aligned is not None:
                            resolved[record_id] = (
                                previous_aligned,
                                "work-exact-text-type-previous-text-aligned",
                            )
                            changed = True
                            continue
                        old_target = next(
                            (
                                sentence
                                for sentence in old_sentences
                                if isinstance(sentence, Mapping)
                                and sentence.get("sentenceId") == old_target_id
                            ),
                            None,
                        )
                        if old_target is not None:
                            aligned_section = _resolve_unique_exact_window_section(
                                bounded, sections, old_sentences
                            )
                            if aligned_section is not None:
                                bounded = [
                                    candidate
                                    for candidate in bounded
                                    if candidate.get("sectionId") == aligned_section
                                ]
                                if len(bounded) == 1:
                                    resolved[record_id] = (
                                        bounded[0],
                                        "work-exact-text-type-window-aligned",
                                    )
                                    changed = True
                                    continue
                            aligned = _resolve_unique_global_neighbor_alignment(
                                bounded,
                                work_id,
                                sentence_index,
                                old_sentences,
                                old_target,
                            )
                            if aligned is not None:
                                resolved[record_id] = (
                                    aligned,
                                    "work-exact-text-type-monotonic-neighbor-aligned",
                                )
                                changed = True

        unresolved = [str(record["id"]) for record in ordered if str(record["id"]) not in resolved]
        if unresolved:
            record_lookup = {str(record["id"]): record for record in ordered}
            details = {}
            for record_id in unresolved[:10]:
                record = record_lookup[record_id]
                source_stem, source_row_index, _ = _record_location(record)
                source_row = source_rows.get((source_stem, source_row_index))
                old_sentences = source_row.get("sentences") if source_row else []
                instance_index = _instance_index(old_sentences) if isinstance(old_sentences, list) else None
                candidate_sections = sorted(
                    {str(value.get("sectionId")) for value in candidates_by_id[record_id]}
                )
                allowed = sorted(clean_label_sections.get((work_id, instance_index), set()))
                details[record_id] = {
                    "candidates": len(candidates_by_id[record_id]),
                    "instanceIndex": instance_index,
                    "candidateSections": candidate_sections[:10],
                    "allowedSections": allowed[:10],
                }
            raise ValueError(
                f"{work_id} 有 {len(unresolved)} 条记录无法用精确文本/句型和单调句序唯一映射: {details}"
            )
        invalid_order: list[dict[str, object]] = []
        by_source_window: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
        for record in ordered:
            source_stem, source_row_index, _ = _record_location(record)
            by_source_window[(source_stem, source_row_index)].append(record)
        for window_records in by_source_window.values():
            window_ordered = sorted(window_records, key=_source_sentence_id)
            old_ids = [_source_sentence_id(record) for record in window_ordered]
            raw_ids = [
                int(resolved[str(record["id"])][0]["sentenceId"])
                for record in window_ordered
            ]
            for left, right, old_left, old_right, raw_left, raw_right in zip(
                window_ordered,
                window_ordered[1:],
                old_ids,
                old_ids[1:],
                raw_ids,
                raw_ids[1:],
            ):
                if (old_left == old_right and raw_left != raw_right) or (
                    old_left < old_right and raw_left >= raw_right
                ):
                    left_id = str(left["id"])
                    right_id = str(right["id"])
                    invalid_order.append(
                        {
                            "leftId": left_id,
                            "rightId": right_id,
                            "oldSentenceIds": [old_left, old_right],
                            "recollectedSentenceIds": [raw_left, raw_right],
                            "methods": [resolved[left_id][1], resolved[right_id][1]],
                        }
                    )
        if invalid_order:
            raise ValueError(
                f"{work_id} 重采集目标映射不保持旧 sentenceId 的严格顺序: "
                f"{invalid_order[:10]}"
            )
    return resolved


def _instance_index(old_sentences: Sequence[object]) -> int | None:
    for sentence in old_sentences:
        if not isinstance(sentence, Mapping):
            continue
        match = re.match(
            r"^Instance index:\s*(\d+)",
            normalize_context_sentence(sentence.get("text")),
        )
        if match is not None:
            return int(match.group(1))
    return None


def _resolve_by_clean_manifest_instance(
    candidates: Sequence[Mapping[str, object]],
    work_id: str,
    old_sentences: Sequence[object],
    clean_label_sections: Mapping[tuple[str, int], set[str]],
) -> Mapping[str, object] | None:
    instance_index = _instance_index(old_sentences)
    if instance_index is None:
        return None
    allowed = clean_label_sections.get((work_id, instance_index), set())
    matched = [
        candidate for candidate in candidates if str(candidate.get("sectionId")) in allowed
    ]
    return matched[0] if len(matched) == 1 else None


def _resolve_by_instance_header(
    candidates: Sequence[Mapping[str, object]],
    sections: Mapping[str, Sequence[Mapping[str, object]]],
    old_sentences: Sequence[object],
) -> Mapping[str, object] | None:
    instance_index = _instance_index(old_sentences)
    if instance_index is None:
        return None
    expected = str(instance_index)
    matched: list[Mapping[str, object]] = []
    for candidate in candidates:
        section = sections[str(candidate["sectionId"])]
        position = next(
            index
            for index, sentence in enumerate(section)
            if sentence is candidate
            or sentence.get("sentenceId") == candidate.get("sentenceId")
        )
        latest_header: str | None = None
        for sentence in section[: position + 1]:
            match = re.match(
                r"^Instance index:\s*(\d+)",
                normalize_context_sentence(sentence.get("text")),
            )
            if match is not None:
                latest_header = match.group(1)
        if latest_header == expected:
            matched.append(candidate)
    return matched[0] if len(matched) == 1 else None


def _resolve_by_exact_previous_text(
    candidates: Sequence[Mapping[str, object]],
    work_id: str,
    previous_text: str,
    sentence_index: Mapping[tuple[str, int], Mapping[str, object]],
    *,
    maximum_span: int = 8,
) -> Mapping[str, object] | None:
    if not previous_text:
        return None
    matched: list[Mapping[str, object]] = []
    for candidate in candidates:
        parts: list[str] = []
        candidate_id = int(candidate["sentenceId"])
        for distance in range(1, maximum_span + 1):
            previous = sentence_index.get((work_id, candidate_id - distance))
            if previous is None or previous.get("sectionId") != candidate.get("sectionId"):
                break
            parts.insert(0, normalize_context_sentence(previous.get("text")))
            combined = normalize_context_sentence("".join(parts))
            if combined == previous_text:
                matched.append(candidate)
                break
            if len(combined) >= len(previous_text):
                break
    return matched[0] if len(matched) == 1 else None


def _find_exact_concatenated_spans(
    work_id: str,
    target_text: str,
    target_type: str,
    sentence_index: Mapping[tuple[str, int], Mapping[str, object]],
    *,
    maximum_span: int = 8,
) -> list[Mapping[str, object]]:
    """Find clean-source line-wrap fragments whose exact concatenation is the target."""

    work_sentences = sorted(
        (
            sentence
            for (candidate_work, _), sentence in sentence_index.items()
            if candidate_work == work_id
        ),
        key=lambda sentence: int(sentence["sentenceId"]),
    )
    matches: list[Mapping[str, object]] = []
    for start, first in enumerate(work_sentences):
        section_id = first["sectionId"]
        combined = ""
        span: list[Mapping[str, object]] = []
        for sentence in work_sentences[start : start + maximum_span]:
            if sentence["sectionId"] != section_id:
                break
            span.append(sentence)
            combined = normalize_context_sentence(
                combined + normalize_context_sentence(sentence.get("text"))
            )
            if combined == target_text and len(span) > 1:
                matches.append(
                    {
                        "sentenceId": first["sentenceId"],
                        "sectionId": section_id,
                        "chapterIndex": first["chapterIndex"],
                        "lineIndex": first["lineIndex"],
                        "text": target_text,
                        "sentenceType": target_type,
                        "sourceOffset": first["sourceOffset"],
                        "spanSentenceIds": [value["sentenceId"] for value in span],
                    }
                )
                break
            if len(combined) >= len(target_text):
                break
    return matches


def _resolve_unique_global_neighbor_alignment(
    candidates: Sequence[Mapping[str, object]],
    work_id: str,
    sentence_index: Mapping[tuple[str, int], Mapping[str, object]],
    old_sentences: Sequence[object],
    old_target: Mapping[str, object],
) -> Mapping[str, object] | None:
    old_position = next(
        index for index, sentence in enumerate(old_sentences) if sentence is old_target
    )
    remaining = list(candidates)
    for radius in range(1, len(old_sentences)):
        for direction in (-1, 1):
            old_index = old_position + direction * radius
            if not 0 <= old_index < len(old_sentences):
                continue
            reference = old_sentences[old_index]
            if not isinstance(reference, Mapping):
                continue
            filtered = []
            for candidate in remaining:
                raw_id = int(candidate["sentenceId"]) + direction * radius
                raw_neighbor = sentence_index.get((work_id, raw_id))
                if raw_neighbor is not None and _same_text_and_type(raw_neighbor, reference):
                    filtered.append(candidate)
            if len(filtered) == 1:
                return filtered[0]
            if filtered:
                remaining = filtered
    return None


def _materialize_record(
    record: Mapping[str, object],
    sentence_index: Mapping[tuple[str, int], Mapping[str, object]],
    sections: Mapping[str, Sequence[Mapping[str, object]]],
    exact_index: Mapping[tuple[str, str, str], Sequence[Mapping[str, object]]],
    source_rows: Mapping[tuple[str, int], Mapping[str, object]],
    source_resolution: Mapping[str, tuple[Mapping[str, object], str]],
    tokenizer,
    *,
    max_length: int,
) -> tuple[dict[str, object], dict[str, object]]:
    record_id = str(record["id"])
    work_id = str(record["workId"])
    source_sentence_id = _source_sentence_id(record)
    source_stem, source_row_index, _ = _record_location(record)
    source_row = source_rows.get((source_stem, source_row_index))
    if source_row is None or not isinstance(source_row.get("sentences"), list):
        raise ValueError(f"缺少旧标注源行: {record_id}")
    old_sentences = source_row["sentences"]
    old_matches = [
        sentence
        for sentence in old_sentences
        if isinstance(sentence, dict)
        and sentence.get("sentenceId") == source_sentence_id
        and normalize_context_sentence(sentence.get("text"))
        == normalize_context_sentence(record.get("text"))
        and sentence.get("type") == record.get("sentenceType")
    ]
    if len(old_matches) != 1:
        raise ValueError(f"旧标注的 sentenceId/text/type 无法严格核验: {record_id}")

    pre_resolved = source_resolution.get(record_id)
    source = pre_resolved[0] if pre_resolved is not None else sentence_index.get((work_id, source_sentence_id))
    mapping_method = pre_resolved[1] if pre_resolved is not None else "global-sentence-id"
    if (
        source is None
        or source.get("text") != record.get("text")
        or source.get("sentenceType") != record.get("sentenceType")
    ):
        if work_id not in _PSEUDO_WORKS:
            raise ValueError(
                f"原文目标文本或句型不一致: {record_id}; "
                f"source={None if source is None else source.get('text')!r}; "
                f"label={record.get('text')!r}"
            )
        exact_matches = list(
            exact_index.get(
                (
                    work_id,
                    normalize_context_sentence(record.get("text")),
                    str(record.get("sentenceType")),
                ),
                (),
            )
        )
        if len(exact_matches) > 1:
            aligned_section = _resolve_unique_exact_window_section(
                exact_matches, sections, old_sentences
            )
            if aligned_section is not None:
                exact_matches = [
                    sentence
                    for sentence in exact_matches
                    if sentence.get("sectionId") == aligned_section
                ]
                mapping_method = "clean-work-exact-window-section"
        if len(exact_matches) > 1:
            aligned = _resolve_unique_exact_neighbor_alignment(
                exact_matches, sections, old_sentences, old_matches[0]
            )
            if aligned is not None:
                exact_matches = [aligned]
                mapping_method = "clean-work-exact-text-type-neighbor-aligned"
        if len(exact_matches) != 1:
            raise ValueError(
                f"clean 原文内目标文本/句型不是唯一精确匹配: {record_id}; "
                f"matches={len(exact_matches)}"
            )
        source = exact_matches[0]
        if mapping_method == "global-sentence-id":
            mapping_method = "clean-work-exact-text-type"
    section_id = str(source["sectionId"])
    section = list(sections[section_id])
    span_sentence_ids = source.get("spanSentenceIds")
    if isinstance(span_sentence_ids, list) and span_sentence_ids:
        span_set = set(span_sentence_ids)
        span_positions = [
            index for index, sentence in enumerate(section) if sentence["sentenceId"] in span_set
        ]
        if (
            len(span_positions) != len(span_sentence_ids)
            or span_positions != list(range(span_positions[0], span_positions[-1] + 1))
        ):
            raise ValueError(f"重采集合并 span 不连续: {record_id}")
        source = dict(source)
        section = (
            section[: span_positions[0]]
            + [source]
            + section[span_positions[-1] + 1 :]
        )
    recollected_sentence_id = source["sentenceId"]
    selection = select_target_context(
        section,
        recollected_sentence_id,
        lambda text: _count_tokens(tokenizer, text),
        max_length=max_length,
    )
    selected = [section[position] for position in selection.selected_positions]
    output = {
        "schema": TARGET_CONTEXT_SCHEMA,
        "schemaVersion": TARGET_CONTEXT_SCHEMA_VERSION,
        "id": record_id,
        "split": record["split"],
        "workId": work_id,
        "sectionId": section_id,
        "sentences": [
            {
                "sentenceId": sentence["sentenceId"],
                "sectionId": section_id,
                "lineIndex": sentence["lineIndex"],
                "text": sentence["text"],
                "sentenceType": sentence["sentenceType"],
            }
            for sentence in selected
        ],
        "targetSentenceId": recollected_sentence_id,
        "sourceKind": record["sourceKind"],
        "source": record["source"],
        "licenseId": record["licenseId"],
        "licenseStatus": record["licenseStatus"],
        "emotions": record["emotions"],
    }
    target_position = next(
        index for index, sentence in enumerate(section) if sentence is source
    )
    eligible_next = (
        section[target_position + 1]
        if target_position + 1 < len(section)
        and section[target_position + 1]["lineIndex"] == source["lineIndex"]
        else None
    )
    audit = {
        "schema": AUDIT_SCHEMA,
        "schemaVersion": 1,
        "id": record_id,
        "split": record["split"],
        "workId": work_id,
        "sectionId": section_id,
        "chapterIndex": source["chapterIndex"],
        "sourceSentenceId": source_sentence_id,
        "recollectedSentenceId": recollected_sentence_id,
        "recollectedSentenceIds": (
            list(span_sentence_ids)
            if isinstance(span_sentence_ids, list)
            else [recollected_sentence_id]
        ),
        "mappingMethod": mapping_method,
        "sourceLineIndex": source["lineIndex"],
        "sourceOffset": source["sourceOffset"],
        "sectionSentenceCount": len(section),
        "availablePreviousSentenceCount": target_position,
        "eligibleNextSentenceId": (
            eligible_next["sentenceId"] if eligible_next is not None else None
        ),
        "selectedSentenceIds": list(selection.selected_sentence_ids),
        "selectedSentenceCount": len(selection.selected_sentence_ids),
        "previousSentenceCount": selection.previous_sentence_count,
        "nextSentenceIncluded": selection.next_sentence_included,
        "renderedText": selection.rendered_text,
        "inputTokenCount": selection.input_token_count,
        "targetTokenCount": selection.target_token_count,
        "contextLimited": selection.context_limited,
        "targetTruncated": selection.target_truncated,
        "stopReason": selection.stop_reason,
        "inputSha256": _canonical_sha256(record),
        "outputSha256": _canonical_sha256(output),
    }
    if output["emotions"] != record["emotions"]:
        raise AssertionError(f"上下文重采集意外改变 emotions: {record_id}")
    return output, audit


def prepare_emotion_target_context512(
    input_dir: str | Path,
    project_root: str | Path,
    output_dir: str | Path,
    *,
    base_model: str | Path = "hfl/chinese-macbert-base",
    max_length: int = CONTEXT_MAX_LENGTH,
) -> dict[str, object]:
    """Build the fixed novel-only r2 target-context training snapshot."""

    if max_length != CONTEXT_MAX_LENGTH:
        raise ValueError(f"目标上下文 ABI 固定 maxLength={CONTEXT_MAX_LENGTH}")
    root = Path(project_root).resolve()
    target = Path(output_dir)
    if target.exists():
        raise ValueError(f"输出目录已存在，拒绝覆盖: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("重采集需要本地 transformers tokenizer") from exc

    records_by_split, input_hashes = _load_novel_records(Path(input_dir))
    work_ids = sorted(
        {str(record["workId"]) for records in records_by_split.values() for record in records}
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model), local_files_only=True, use_fast=True
    )
    special_token_ids = ensure_emotion_special_tokens(tokenizer)
    source_rows, source_windows = _load_source_rows(
        records_by_split,
        root / "speaker-id" / "typescript" / "data",
    )
    clean_label_sections, clean_label_manifests = _load_clean_label_sections(
        root, work_ids
    )

    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        audit_dir = temporary / "audit"
        audit_dir.mkdir(parents=True)
        sections_path = audit_dir / "source-sections.jsonl"
        source_manifest = collect_source_sections(root, work_ids, sections_path)
        sentence_index, sections, exact_index = _load_sections(sections_path)
        source_resolution = _resolve_recollected_sources(
            records_by_split,
            sentence_index,
            exact_index,
            sections,
            source_rows,
            clean_label_sections,
        )

        output_by_split: dict[str, list[dict[str, object]]] = {}
        audits: list[dict[str, object]] = []
        token_counts: list[int] = []
        context_counts: Counter[str] = Counter()
        target_types: Counter[str] = Counter()
        selected_types: Counter[str] = Counter()
        mapping_methods: Counter[str] = Counter()
        stop_reasons: Counter[str] = Counter()
        next_eligible = 0
        next_included = 0
        for split in ("train", "dev", "test"):
            outputs: list[dict[str, object]] = []
            for record in records_by_split[split]:
                output, audit = _materialize_record(
                    record,
                    sentence_index,
                    sections,
                    exact_index,
                    source_rows,
                    source_resolution,
                    tokenizer,
                    max_length=max_length,
                )
                outputs.append(output)
                audits.append(audit)
                token_counts.append(int(audit["inputTokenCount"]))
                context_counts[str(audit["selectedSentenceCount"])] += 1
                stop_reasons[str(audit["stopReason"])] += 1
                target_types[str(record["sentenceType"])] += 1
                selected_types.update(
                    str(sentence["sentenceType"]) for sentence in output["sentences"]
                )
                mapping_methods[str(audit["mappingMethod"])] += 1
                next_eligible += int(audit["eligibleNextSentenceId"] is not None)
                next_included += int(bool(audit["nextSentenceIncluded"]))
            output_by_split[split] = outputs

        counts = {split: len(records) for split, records in output_by_split.items()}
        if counts != EXPECTED_NOVEL_SPLITS:
            raise ValueError(
                f"目标上下文 split 计数不匹配: expected={EXPECTED_NOVEL_SPLITS}, actual={counts}"
            )
        for split, records in output_by_split.items():
            _write_jsonl(temporary / f"{split}.jsonl", records)
        _write_jsonl(audit_dir / "context-selection.jsonl", audits)
        _write_jsonl(
            audit_dir / "target-truncation-exceptions.jsonl",
            (record for record in audits if record["targetTruncated"]),
        )
        report = {
            "schema": TARGET_CONTEXT_SCHEMA,
            "schemaVersion": TARGET_CONTEXT_SCHEMA_VERSION,
            "trainingStarted": False,
            "recordCount": sum(counts.values()),
            "splitCounts": counts,
            "workCount": len(work_ids),
            "conditioningAbi": EMOTION_ABI,
            "contextPolicy": CONTEXT_POLICY,
            "contextEncoding": CONTEXT_ENCODING,
            "maxLength": max_length,
            "specialTokens": list(EMOTION_SPECIAL_TOKENS),
            "specialTokenIds": special_token_ids,
            "contextSentenceCounts": dict(sorted(context_counts.items())),
            "sameLineNextEligibleCount": next_eligible,
            "sameLineNextIncludedCount": next_included,
            "sameLineNextInclusionRate": next_included / next_eligible if next_eligible else 0.0,
            "targetSentenceTypes": dict(sorted(target_types.items())),
            "selectedSentenceTypes": dict(sorted(selected_types.items())),
            "mappingMethods": dict(sorted(mapping_methods.items())),
            "tokenCounts": {
                "min": min(token_counts),
                "p50": _percentile(token_counts, 0.50),
                "p95": _percentile(token_counts, 0.95),
                "p99": _percentile(token_counts, 0.99),
                "max": max(token_counts),
            },
            "stopReasons": dict(sorted(stop_reasons.items())),
            "targetTruncatedCount": sum(bool(item["targetTruncated"]) for item in audits),
            "labelsUnchanged": True,
        }
        (temporary / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        material_files = [
            "train.jsonl",
            "dev.jsonl",
            "test.jsonl",
            "report.json",
            "audit/source-collection-spec.json",
            "audit/source-sections.jsonl",
            "audit/source-sections.jsonl.manifest.json",
            "audit/context-selection.jsonl",
            "audit/target-truncation-exceptions.jsonl",
        ]
        manifest = {
            "schema": TARGET_CONTEXT_SCHEMA,
            "schemaVersion": TARGET_CONTEXT_SCHEMA_VERSION,
            "status": "completed",
            "parentDataset": Path(input_dir).resolve().as_posix(),
            "inputSha256": input_hashes,
            "sourceCollection": source_manifest,
            "sourceAnnotationWindows": source_windows,
            "cleanLabelManifests": clean_label_manifests,
            "baseModel": str(base_model),
            "tokenizerVocabularySize": len(tokenizer),
            "specialTokenIds": special_token_ids,
            "conditioningAbi": EMOTION_ABI,
            "contextPolicy": CONTEXT_POLICY,
            "contextEncoding": CONTEXT_ENCODING,
            "maxLength": max_length,
            "splitCounts": counts,
            "recordCount": sum(counts.values()),
            "fileSha256": {name: _sha256_file(temporary / name) for name in material_files},
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report | {
        "output": target.resolve().as_posix(),
        "manifestSha256": _sha256_file(target / "manifest.json"),
    }
