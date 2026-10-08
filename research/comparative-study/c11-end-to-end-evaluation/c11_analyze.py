#!/usr/bin/env python3
"""C11 evidence aggregator. Standard-library only; never edits raw inputs."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, platform, random, statistics, sys
from datetime import datetime, timezone
from pathlib import Path

DATASETS = ('SCI', 'NFC')
ARMS = tuple('ABCDEFGHI')
METRICS = ('recall10_qrels', 'ndcg10_qrels', 'latency_us')
SEED = 20261011

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def mean(xs): return statistics.fmean(xs) if xs else None
def percentile(xs, p):
    if not xs: return None
    xs = sorted(xs)
    return xs[min(len(xs)-1, max(0, math.ceil(p * len(xs))-1))]

def row_mrr(row):
    relevant = set(str(x) for x in row.get('relevant_ids', []))
    for rank, hit in enumerate(row.get('hits', []), 1):
        if str(hit) in relevant: return 1.0 / rank
    return 0.0

def row_key(row, index):
    for key in ('query_id', 'qid', 'query_id_index', 'query'):
        if key in row: return str(row[key])
    return f'position:{index}'

def read_per_query(run_dir: Path, arm: str, repetitions: int):
    out = {}
    artifacts = run_dir / 'artifacts'
    for rep in range(1, repetitions + 1):
        path = artifacts / f'ARM-{arm}-rep{rep}.json'
        if not path.exists(): continue
        payload = json.loads(path.read_text(encoding='utf-8'))
        rows = payload.get('per_query', [])
        for i, row in enumerate(rows):
            key = (rep, row_key(row, i))
            out[key] = {
                'recall10_qrels': row.get('recall10_qrels'),
                'ndcg10_qrels': row.get('ndcg10_qrels'),
                'mrr10_qrels': row_mrr(row),
                'latency_us': row.get('latency_us'),
            }
    return out

def paired_bootstrap(a, b, metric, draws=10000, seed=SEED):
    keys = sorted(set(a) & set(b))
    pairs = [(a[k].get(metric), b[k].get(metric)) for k in keys]
    pairs = [(float(x), float(y)) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 2: return {'n_paired': len(pairs), 'delta_mean': None, 'ci95_low': None, 'ci95_high': None}
    diffs = [x-y for x, y in pairs]
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        samples.append(statistics.fmean([diffs[rng.randrange(len(diffs))] for _ in diffs]))
    return {'n_paired': len(diffs), 'delta_mean': statistics.fmean(diffs),
            'ci95_low': percentile(samples, .025), 'ci95_high': percentile(samples, .975),
            'method': 'paired query-repetition bootstrap; percentile interval; no multiplicity correction',
            'draws': draws, 'seed': seed}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True, help='Comparative-study raw directory')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--c10-jsonl', type=Path, help='Optional C10 JSONL; kept as a separate microbenchmark track')
    parser.add_argument('--require-complete', action='store_true', help='Fail unless all datasets/arms have five reps and provenance')
    args = parser.parse_args()
    raw_root = args.raw_root.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    query_data = {}
    issues = []
    for metrics_path in sorted(raw_root.glob('*/metrics.json')):
        try: m = json.loads(metrics_path.read_text(encoding='utf-8'))
        except Exception as exc:
            issues.append(f'{metrics_path}: invalid JSON: {exc}'); continue
        dataset = str(m.get('dataset', '')).upper()
        if dataset in ('SCIFACT', 'DS-SCIFACT', 'SCI-FACT'): dataset = 'SCI'
        if dataset in ('NFCORPUS', 'DS-NFCORPUS'): dataset = 'NFC'
        arm = str(m.get('mode', '')).replace('MODE-', '').upper()
        split = str(m.get('split', '')).upper()
        if dataset not in DATASETS or arm not in ARMS: continue
        if split != 'TEST':
            continue
        if m.get('union_identity_ok') is False: issues.append(f'{metrics_path}: union_identity_ok=false')
        if m.get('union_identity_ok') is None: issues.append(f'{metrics_path}: union_identity_ok missing')
        reps = m.get('per_rep', [])
        rep_count = int(m.get('rep_count', len(reps) or 0))
        if rep_count < 5: issues.append(f'{metrics_path}: only {rep_count} repetitions (required 5)')
        for field in ('test_queries', 'candidate_budget', 'ef_search', 'union_identity_ok'):
            if field not in m: issues.append(f'{metrics_path}: missing {field}')
        env_path = metrics_path.parent / 'artifacts' / 'environment.yaml'
        if not env_path.exists(): issues.append(f'{metrics_path}: missing artifacts/environment.yaml')
        summary = {}
        for metric in ('recall10_qrels_mean', 'ndcg10_qrels_mean', 'mrr10_qrels_mean', 'p50_us', 'p90_us', 'p95_us', 'deadline_exceeded'):
            vals = [r.get(metric) for r in reps if isinstance(r, dict) and r.get(metric) is not None]
            summary[metric] = mean([float(v) for v in vals]) if vals else None
        run_id = m.get('run_id', metrics_path.parent.name)
        records.append({'run_id': run_id, 'dataset': dataset, 'arm': arm, 'split': split,
                        'rep_count': rep_count, 'test_queries': m.get('test_queries'),
                        'candidate_budget': m.get('candidate_budget'), 'ef_search': m.get('ef_search'),
                        'union_identity_ok': m.get('union_identity_ok'), 'summary': summary,
                        'metrics_path': str(metrics_path.relative_to(raw_root)), 'metrics_sha256': sha256(metrics_path)})
        query_data[(dataset, arm)] = read_per_query(metrics_path.parent, arm, rep_count)
    by_cell = {(r['dataset'], r['arm']): r for r in records}
    expected = {(d, a) for d in DATASETS for a in ARMS}
    missing = sorted(expected - set(by_cell))
    if missing: issues.append('missing TEST cells: ' + ', '.join(f'{d}/{a}' for d,a in missing))
    contrasts = []
    for dataset in DATASETS:
        for arm, baseline in [('B','A')] + [(a,'B') for a in 'CDEFGHI']:
            left = query_data.get((dataset, arm), {})
            right = query_data.get((dataset, baseline), {})
            for metric in ('recall10_qrels', 'ndcg10_qrels', 'mrr10_qrels', 'latency_us'):
                boot = paired_bootstrap(left, right, metric, seed=SEED + ord(arm) + ord(dataset[0]))
                contrasts.append({'dataset': dataset, 'contrast': f'{arm}-{baseline}', 'metric': metric, **boot})
    c10 = []
    if args.c10_jsonl:
        for line_no, line in enumerate(args.c10_jsonl.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip(): continue
            try: c10.append(json.loads(line))
            except Exception as exc: issues.append(f'C10 JSONL line {line_no} invalid: {exc}')
    provenance = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
                  'platform': platform.platform(), 'raw_root': str(raw_root), 'c11_analyzer_sha256': sha256(Path(__file__)),
                  'raw_metrics': [{'path': r['metrics_path'], 'sha256': r['metrics_sha256']} for r in records],
                  'c10_jsonl': str(args.c10_jsonl.resolve()) if args.c10_jsonl else None,
                  'c10_jsonl_sha256': sha256(args.c10_jsonl) if args.c10_jsonl else None}
    result = {'schema_version': '1.0.0', 'empirical_completion': not missing and not issues,
              'required_cells': len(expected), 'observed_cells': len(by_cell), 'missing_cells': missing,
              'issues': issues, 'records': records, 'contrasts': contrasts,
              'c10_cache_microbenchmarks_separate_track': c10, 'provenance': provenance}
    (out_dir / 'c11_results.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    with (out_dir / 'c11_summary.csv').open('w', newline='', encoding='utf-8') as f:
        fields = ['run_id','dataset','arm','split','rep_count','test_queries','candidate_budget','ef_search','union_identity_ok',
                  'recall10_qrels_mean','ndcg10_qrels_mean','mrr10_qrels_mean','p50_us','p90_us','p95_us','deadline_exceeded','metrics_path','metrics_sha256']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for r in records: writer.writerow({**{k:r.get(k) for k in fields if k not in ('recall10_qrels_mean','ndcg10_qrels_mean','mrr10_qrels_mean','p50_us','p90_us','p95_us','deadline_exceeded')}, **r['summary']})
    report = ['# C11 End-to-End Evaluation Results', '', f"Generated: {provenance['generated_utc']}", '',
              f"Evidence status: {'COMPLETE' if result['empirical_completion'] else 'INCOMPLETE — no paper-level claim should be made'}",
              f"Required TEST cells: {len(expected)}; observed: {len(by_cell)}.", '',
              '## Per-arm results', '', '| Dataset | Arm | Reps | Queries | Recall@10 | nDCG@10 | MRR@10 | p50 µs | p90 µs | p95 µs | Deadlines |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted(records, key=lambda x:(x['dataset'],x['arm'])):
        s=r['summary']; fmt=lambda v: 'NA' if v is None else f'{v:.6g}'
        report.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(r['dataset'],r['arm'],r['rep_count'],r['test_queries'],fmt(s['recall10_qrels_mean']),fmt(s['ndcg10_qrels_mean']),fmt(s['mrr10_qrels_mean']),fmt(s['p50_us']),fmt(s['p90_us']),fmt(s['p95_us']),fmt(s['deadline_exceeded'])))
    report += ['', '## Paired contrasts', '', 'Positive quality deltas favor the first arm. Negative latency deltas favor the first arm. CIs are unavailable if per-query pairing is incomplete.',
               '', '| Dataset | Contrast | Metric | n paired | Mean delta | 95% CI |', '|---|---|---|---:|---:|---|']
    for c in contrasts:
        fmt=lambda v: 'NA' if v is None else f'{v:.6g}'
        report.append(f"| {c['dataset']} | {c['contrast']} | {c['metric']} | {c['n_paired']} | {fmt(c['delta_mean'])} | [{fmt(c['ci95_low'])}, {fmt(c['ci95_high'])}] |")
    report += ['', '## Validation issues', ''] + ([f'- {i}' for i in issues] if issues else ['- None detected by the aggregator.'])
    report += ['', '## Interpretation limits', '', '- C10 cache lifecycle timings are a separate microbenchmark and are not end-to-end query latency.',
               '- These in-repository controls are not claims of superiority over external databases.',
               '- CI validates code and fixtures; only frozen, provenance-complete TEST runs provide empirical evidence.',
               '- No performance or quality conclusion is asserted by this generated report.', '']
    (out_dir / 'C11_REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'required_cells': len(expected), 'observed_cells': len(by_cell), 'missing_cells': missing, 'issues': issues, 'output_dir': str(out_dir)}, indent=2))
    if args.require_complete and (missing or issues): return 2
    return 0

if __name__ == '__main__': sys.exit(main())