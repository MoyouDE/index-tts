import hashlib
import json

from indextts.emotion.coverage import COVERAGE_SCHEMA, build_coverage_backlog


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_coverage_backlog_excludes_existing_candidates_and_preserves_sources(tmp_path):
    source = _jsonl(
        tmp_path / "speaker" / "book_annotated.jsonl",
        [
            {
                "targets": [2],
                "sentences": [
                    {
                        "sentenceId": 1,
                        "text": "他感到恶心和反感。",
                        "type": "narration",
                        "lineIndex": 1,
                    },
                    {
                        "sentenceId": 2,
                        "text": "真是令人作呕！",
                        "type": "dialogue",
                        "lineIndex": 2,
                    },
                    {
                        "sentenceId": 3,
                        "text": "他黯然地站在凄凉的街上。",
                        "type": "narration",
                        "lineIndex": 3,
                    },
                ],
            }
        ],
    )
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    existing = _jsonl(
        tmp_path / "existing.jsonl",
        [{"id": "emotion-candidate-book-r000001-s000001"}],
    )
    report = build_coverage_backlog(
        source.parent,
        existing,
        tmp_path / "coverage",
        quotas={"disgusted": 2, "melancholic": 2},
        license_status="test-only",
    )

    assert report["selectedByEmotion"] == {"disgusted": 2, "melancholic": 1}
    assert report["uniqueSelected"] == 2
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    rows = [
        json.loads(line)
        for line in (tmp_path / "coverage" / "coverage-candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["schema"] for row in rows} == {COVERAGE_SCHEMA}
    assert {tuple(row["coverageTargets"]) for row in rows} == {
        ("disgusted",),
        ("disgusted", "melancholic"),
    }
    assert all("emotions" not in row for row in rows)
    blind_rows = [
        json.loads(line)
        for line in (tmp_path / "coverage" / "annotation-candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["schema"] for row in blind_rows} == {"readest-emotion-candidate-v1"}
    assert all(
        not {"coverageTargets", "coverageEvidence", "coverageScore"}.intersection(row)
        for row in blind_rows
    )
