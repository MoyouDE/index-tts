"""Extract emotion-label candidates from the existing speaker annotations.

The source JSONL files are treated as immutable.  This module deliberately
produces a separate candidate schema instead of pretending that speaker
annotations are emotion labels.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


CANDIDATE_SCHEMA = "readest-emotion-candidate-v1"
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
_SPACE = re.compile(r"\s+")
_METADATA = re.compile(
    r"^(?:第.{1,16}[章节回卷部]|chapter\s*\d+|番外|正文完|"
    r"拥有(?:特质|奖励|物品)|当前(?:等级|状态)|系统(?:提示|公告)|"
    r"任务(?:目标|奖励)|属性(?:面板|列表))(?::|：|$).*$",
    re.I,
)
_DIFFICULTY_PATTERNS = {
    "negation": re.compile(r"不|没|无|未|别|莫|难道|却|但|然而|反而|只是|偏偏|哪知|怎料|没想到|原来"),
    "sarcasm": re.compile(r"呵|哼|冷笑|讥|嘲|讽|笑死|可笑|真是"),
    "suspense": re.compile(r"忽然|突然|神秘|诡异|黑暗|背后|危机|危险|杀机|不知|谜|悬"),
    "fright": re.compile(r"恐|害怕|颤|发抖|惊叫|尖叫|逃|心惊|骇"),
    "inner_monologue": re.compile(r"心想|暗道|心中|内心|想着|念头|他想|她想"),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def normalize_text(value: object) -> str:
    text = _ZERO_WIDTH.sub("", str(value or "")).strip()
    return _SPACE.sub(" ", text)


def _is_metadata(text: str) -> bool:
    return bool(_METADATA.fullmatch(text))


def _previous_sentence(sentences: list[dict[str, object]], index: int) -> str:
    for previous in reversed(sentences[:index]):
        text = normalize_text(previous.get("text"))
        if text:
            return text
    return ""


@dataclass
class _Selected:
    score: int
    key: tuple[str, int, str]
    record: dict[str, object]
    source_rows: list[int] = field(default_factory=list)


class _StratumSelector:
    """Keep the deterministic lowest-hash records for one work/stratum."""

    def __init__(self, quota: int, seed: int):
        self.quota = max(0, int(quota))
        self.seed = int(seed)
        self.items: dict[tuple[str, int, str], _Selected] = {}
        self.heap: list[tuple[int, tuple[str, int, str]]] = []

    def _score(self, key: tuple[str, int, str]) -> int:
        payload = f"{self.seed}|{key[0]}|{key[1]}|{key[2]}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    @staticmethod
    def _prefer(new: dict[str, object], old: dict[str, object]) -> bool:
        new_previous = normalize_text(new.get("previousText"))
        old_previous = normalize_text(old.get("previousText"))
        return (len(new_previous), len(normalize_text(new.get("text")))) > (
            len(old_previous),
            len(normalize_text(old.get("text"))),
        )

    def add(self, key: tuple[str, int, str], record: dict[str, object], source_row: int) -> None:
        if self.quota == 0:
            return
        existing = self.items.get(key)
        if existing is not None:
            if source_row not in existing.source_rows:
                existing.source_rows.append(source_row)
            if self._prefer(record, existing.record):
                record["sourceRowIndices"] = existing.source_rows
                existing.record = record
            return
        score = self._score(key)
        if len(self.items) >= self.quota:
            while self.heap and self.heap[0][1] not in self.items:
                heapq.heappop(self.heap)
            if self.heap and score >= -self.heap[0][0]:
                return
            if self.heap:
                _, evicted_key = heapq.heappop(self.heap)
                self.items.pop(evicted_key, None)
        selected = _Selected(score, key, record, [source_row])
        record["sourceRowIndices"] = selected.source_rows
        self.items[key] = selected
        heapq.heappush(self.heap, (-score, key))

    def records(self) -> list[dict[str, object]]:
        return [item.record for item in sorted(self.items.values(), key=lambda item: item.score)]


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "-", value).strip("-")
    return slug or "work"


def _candidate_record(
    *,
    source_path: Path,
    source_sha256: str,
    work_id: str,
    source_row: int,
    sentence: dict[str, object],
    previous_text: str,
    is_dialogue_target: bool,
) -> tuple[tuple[str, int, str], dict[str, object]] | None:
    text = normalize_text(sentence.get("text"))
    if not text or _is_metadata(text):
        return None
    sentence_id = sentence.get("sentenceId")
    try:
        sentence_id = int(sentence_id)
    except (TypeError, ValueError):
        return None
    sentence_type = normalize_text(sentence.get("type"))
    if sentence_type not in {"dialogue", "narration"}:
        return None
    text_hash = sha256_text(text)
    previous_hash = sha256_text(previous_text)
    key = (work_id, sentence_id, text_hash)
    record = {
        "schema": CANDIDATE_SCHEMA,
        "id": f"emotion-candidate-{_safe_slug(work_id)}-r{source_row:06d}-s{sentence_id:06d}",
        "workId": work_id,
        "sourceFile": source_path.name,
        "sourceRowIndex": source_row,
        "sourceSentenceId": sentence_id,
        "previousText": previous_text,
        "text": text,
        "sentenceType": sentence_type,
        "lineIndex": sentence.get("lineIndex"),
        "isDialogueTarget": bool(is_dialogue_target),
        "source": f"speaker-id:{source_path.stem}",
        "sourceSha256": source_sha256,
        "textSha256": text_hash,
        "previousTextSha256": previous_hash,
        "licenseStatus": "pending",
        "licenseEvidence": None,
    }
    return key, record


def _work_quotas(total: int, works: list[str], ratio: float) -> dict[str, int]:
    requested = int(round(total * ratio))
    base, remainder = divmod(requested, max(1, len(works)))
    return {work: base + (index < remainder) for index, work in enumerate(works)}


def extract_candidates(
    input_root: str | Path,
    output_dir: str | Path,
    *,
    pool_size: int = 20_000,
    seed: int = 20260829,
    license_status: str = "pending",
) -> dict[str, object]:
    """Extract a deterministic, balanced candidate pool without changing sources."""
    if pool_size < 100:
        raise ValueError("pool_size 至少为 100")
    if license_status not in {"pending", "test-only"}:
        raise ValueError("license_status 只能是 pending 或 test-only")
    source_root = Path(input_root)
    target = Path(output_dir)
    files = sorted(source_root.glob("*_annotated.jsonl"))
    if not files:
        raise ValueError(f"未找到 *_annotated.jsonl: {source_root}")
    works = [path.stem.removesuffix("_annotated") for path in files]
    dialogue_quotas = _work_quotas(pool_size, works, 0.4)
    narration_quotas = _work_quotas(pool_size, works, 0.6)
    selectors: dict[tuple[str, str], _StratumSelector] = {
        (work, "dialogue"): _StratumSelector(dialogue_quotas[work], seed)
        for work in works
    }
    selectors.update(
        {
            (work, "narration"): _StratumSelector(narration_quotas[work], seed + 1)
            for work in works
        }
    )
    source_manifest: list[dict[str, object]] = []
    total_windows = 0
    total_sentences = 0
    total_targets = 0
    work_stats = {
        work: {
            "dialogue": 0,
            "narration": 0,
            "total": 0,
            "difficultyHeuristics": {name: 0 for name in _DIFFICULTY_PATTERNS},
        }
        for work in works
    }
    for source_path in files:
        source_sha256 = sha256_bytes(source_path.read_bytes())
        work_id = source_path.stem.removesuffix("_annotated")
        source_manifest.append(
            {
                "workId": work_id,
                "sourceFile": source_path.name,
                "sourceSha256": source_sha256,
                "licenseStatus": license_status,
                "licenseEvidence": None,
            }
        )
        with source_path.open("r", encoding="utf-8") as handle:
            for source_row, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                total_windows += 1
                sentences = row.get("sentences", [])
                if not isinstance(sentences, list):
                    continue
                sentence_list = [item for item in sentences if isinstance(item, dict)]
                total_sentences += len(sentence_list)
                target_ids: set[int] = set()
                for value in row.get("targets", []):
                    try:
                        target_ids.add(int(value))
                    except (TypeError, ValueError):
                        continue
                total_targets += len(target_ids)
                for index, sentence in enumerate(sentence_list):
                    try:
                        sentence_id = int(sentence.get("sentenceId"))
                    except (TypeError, ValueError):
                        continue
                    is_target = sentence_id in target_ids
                    sentence_type = normalize_text(sentence.get("type"))
                    if is_target and sentence_type == "dialogue":
                        stratum = "dialogue"
                    elif not is_target and sentence_type == "narration":
                        stratum = "narration"
                    else:
                        continue
                    previous_text = _previous_sentence(sentence_list, index)
                    candidate = _candidate_record(
                        source_path=source_path,
                        source_sha256=source_sha256,
                        work_id=work_id,
                        source_row=source_row,
                        sentence=sentence,
                        previous_text=previous_text,
                        is_dialogue_target=is_target,
                    )
                    if candidate is not None:
                        key, record = candidate
                        record["licenseStatus"] = license_status
                        selectors[(work_id, stratum)].add(key, record, source_row)
    records = [
        record
        for selector in selectors.values()
        for record in selector.records()
    ]
    records.sort(key=lambda record: str(record["id"]))
    difficulty_counts = {name: 0 for name in _DIFFICULTY_PATTERNS}
    for record in records:
        work_stat = work_stats[str(record["workId"])]
        stratum = "dialogue" if record["sentenceType"] == "dialogue" else "narration"
        work_stat[stratum] += 1
        work_stat["total"] += 1
        context = f"{record.get('previousText', '')} {record.get('text', '')}"
        for name, pattern in _DIFFICULTY_PATTERNS.items():
            if pattern.search(context):
                difficulty_counts[name] += 1
                work_stat["difficultyHeuristics"][name] += 1
    target.mkdir(parents=True, exist_ok=True)
    candidates_path = target / "candidates.jsonl"
    with candidates_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    license_path = target / "source-license.json"
    license_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "policy": "license evidence required before formal training",
                "sources": source_manifest,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "schema": CANDIDATE_SCHEMA,
        "seed": seed,
        "requestedPoolSize": pool_size,
        "candidateCount": len(records),
        "dialogueCount": sum(bool(item["isDialogueTarget"]) for item in records),
        "narrationCount": sum(not bool(item["isDialogueTarget"]) for item in records),
        "sourceCount": len(files),
        "sourceWindows": total_windows,
        "sourceSentences": total_sentences,
        "sourceTargets": total_targets,
        "candidateFile": str(candidates_path),
        "candidateFileSha256": sha256_bytes(candidates_path.read_bytes()),
        "licenseFile": str(license_path),
        "licenseStatus": license_status,
        "workStats": work_stats,
        "difficultyHeuristicCounts": difficulty_counts,
    }
    (target / "candidate-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def iter_candidate_records(path: str | Path) -> Iterable[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"候选文件第 {line_number} 行不是对象")
            yield record
