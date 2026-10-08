#!/usr/bin/env python3
"""Validate the frozen C11 matrix without requiring private datasets."""
import csv, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / 'c11-run-plan.csv'
EXPECTED = {(d, a) for d in ('SCI', 'NFC') for a in 'ABCDEFGHI'}

def validate(path=PLAN):
    errors = []
    with path.open(newline='', encoding='utf-8') as f: rows = list(csv.DictReader(f))
    required = {'cell_id','dataset','split','arm','baseline','repetitions','query_count','k','ef_search','candidate_budget','warmup','quality_metrics','latency_metrics','cache_track'}
    if not rows: return ['run plan is empty']
    if not required.issubset(rows[0]): errors.append('missing columns: ' + ', '.join(sorted(required-set(rows[0]))))
    cells = set(); ids = set()
    for n, row in enumerate(rows, 2):
        key = (row.get('dataset'), row.get('arm'))
        if row.get('cell_id') in ids: errors.append(f'line {n}: duplicate cell_id {row.get("cell_id")}')
        ids.add(row.get('cell_id'))
        if key in cells: errors.append(f'line {n}: duplicate dataset/arm {key}')
        cells.add(key)
        if row.get('split') != 'TEST': errors.append(f'line {n}: primary matrix must use TEST split')
        if row.get('repetitions') != '5': errors.append(f'line {n}: repetitions must be 5')
        if row.get('k') != '10' or row.get('ef_search') != '64' or row.get('candidate_budget') != '500' or row.get('warmup') != '20': errors.append(f'line {n}: frozen retrieval settings changed')
        expected_count = '300' if row.get('dataset') == 'SCI' else '323' if row.get('dataset') == 'NFC' else None
        if row.get('query_count') != expected_count: errors.append(f'line {n}: unexpected query count for {row.get("dataset")}')
        expected_baseline = 'NONE' if row.get('arm') == 'A' else 'A' if row.get('arm') == 'B' else 'B'
        if row.get('baseline') != expected_baseline: errors.append(f'line {n}: baseline for arm {row.get("arm")} must be {expected_baseline}')
    missing = EXPECTED - cells; extra = cells - EXPECTED
    if missing: errors.append('missing cells: ' + ', '.join(f'{d}/{a}' for d,a in sorted(missing)))
    if extra: errors.append('unexpected cells: ' + ', '.join(f'{d}/{a}' for d,a in sorted(extra)))
    return errors

if __name__ == '__main__':
    errors = validate()
    if errors:
        print('\n'.join('ERROR: '+e for e in errors)); sys.exit(1)
    print('C11 run plan valid: 18 frozen dataset/arm cells; TEST-only; five repetitions each.')