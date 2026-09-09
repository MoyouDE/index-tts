"""Batch semantic review with immediate, auditable split updates."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIMS = 'happy angry sad afraid disgusted melancholic surprised calm'.split()

def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

def sha(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()

def split_rows():
    result = {}
    for split in ('train', 'dev', 'test'):
        result[split] = read(ROOT / (split + '.jsonl'))
        assert len({r['id'] for r in result[split]}) == len(result[split])
    return result

def original_row(current, emotions):
    row = dict(current)
    row['emotions'] = emotions
    return row

def validate_reviewed(by_id, reviewed):
    applied = True
    for r in reviewed:
        assert r['id'] in by_id
        current = by_id[r['id']]
        assert current['split'] == r['split']
        if current['emotions'] == r['emotions']:
            assert sha(original_row(current, r['previousEmotions'])) == r['inputSha256'], r['id']
        else:
            assert sha(current) == r['inputSha256'], r['id']
            applied = False
    return applied

def pending():
    excluded = {r['id'] for r in read(ROOT / 'corrections.jsonl')}
    ledger = ROOT / 'remaining-review.jsonl'
    reviewed = read(ledger) if ledger.exists() else []
    assert len({r['id'] for r in reviewed}) == len(reviewed)
    done = excluded | {r['id'] for r in reviewed}
    split_data = split_rows()
    rows = [r for split in ('train', 'dev', 'test') for r in split_data[split]]
    assert len({r['id'] for r in rows}) == len(rows)
    by_id = {r['id']: r for r in rows}
    for r in reviewed:
        assert r['id'] not in excluded
    applied = validate_reviewed(by_id, reviewed)
    return [r for r in rows if r['id'] not in done], reviewed, applied

def apply_batch(output):
    updates = {r['id']: r['emotions'] for r in output}
    if not updates:
        return
    split_data = split_rows()
    changed = set()
    for split, rows in split_data.items():
        for row in rows:
            if row['id'] in updates:
                assert row['emotions'] == next(
                    reviewed['previousEmotions'] for reviewed in output if reviewed['id'] == row['id']
                ), row['id']
                row['emotions'] = updates[row['id']]
                changed.add(split)
    assert len(changed) > 0
    for split in changed:
        path = ROOT / (split + '.jsonl')
        temp = path.with_suffix('.jsonl.tmp')
        temp.write_text(
            ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in split_data[split]),
            encoding='utf-8', newline='\n'
        )
        temp.replace(path)

def sync_reviewed(reviewed):
    split_data = split_rows()
    by_id = {row['id']: row for rows in split_data.values() for row in rows}
    unapplied = []
    for review in reviewed:
        row = by_id[review['id']]
        if row['emotions'] == review['emotions']:
            assert sha(original_row(row, review['previousEmotions'])) == review['inputSha256'], review['id']
        else:
            assert sha(row) == review['inputSha256'], review['id']
            unapplied.append(review)
    apply_batch(unapplied)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['show', 'record', 'status', 'sync'])
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--decisions', type=Path)
    args = parser.parse_args()
    rows, reviewed, applied = pending()
    if args.action == 'show':
        for i, r in enumerate(rows[:args.count]):
            print(json.dumps({'index': i, 'id': r['id'], 'target': r['targetSentenceId'], 'context': [[s['sentenceId'], s['sentenceType'], s['text']] for s in r['sentences']]}, ensure_ascii=False))
    elif args.action == 'record':
        decisions = [json.loads(s) for s in sys.stdin if s.strip()] if str(args.decisions) == '-' else read(args.decisions)
        assert decisions and len(decisions) <= len(rows)
        output = []
        for r, d in zip(rows, decisions):
            assert d['id'] == r['id'], (d['id'], r['id'])
            assert len(d['values']) == 8
            assert all(type(v) in (int, float) and 0 <= v <= 1 for v in d['values'])
            assert d['reason'].strip()
            emotions = dict(zip(DIMS, d['values']))
            output.append({'id': r['id'], 'split': r['split'], 'inputSha256': sha(r), 'decision': 'retain' if emotions == r['emotions'] else 'revise', 'previousEmotions': r['emotions'], 'emotions': emotions, 'reason': d['reason'], 'confidence': d.get('confidence', 'high'), 'reviewMethod': 'contextual-semantic-readest-emotion-review', 'targetSentenceId': r['targetSentenceId']})
        apply_batch(output)
        with (ROOT / 'remaining-review.jsonl').open('a', encoding='utf-8', newline='\n') as f:
            for r in output:
                f.write(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n')
        rows, reviewed, applied = pending()
    elif args.action == 'sync':
        sync_reviewed(reviewed)
        rows, reviewed, applied = pending()
    print(json.dumps({'reviewed': len(reviewed), 'remaining': len(rows), 'revised': sum(r['decision'] == 'revise' for r in reviewed), 'appliedToSplits': applied}, ensure_ascii=False))

if __name__ == '__main__':
    main()
