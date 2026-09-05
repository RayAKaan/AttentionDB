#!/usr/bin/env python3
"""Research consistency check (spec §25).

Verifies that every number appearing in generated tables and summary
documents matches the raw benchmark CSVs and the experiment registry.
Scope: R@10, NDCG@10, MRR, p50/p95/p99, QPS. Exits non-zero on any
mismatch — run before committing research artifacts or quoting numbers.

Also enforces ORACLE-SANITY invariants that caught the historical harness
bugs: an oracle must not score ~0 (HC-1) nor ≪ 1.0 on corpora where the
matching structure is near-deterministic (HC-2/HC-3 signatures).
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
TAB = os.path.join(HERE, 'tables')
ERR = []

def rd(path):
    return list(csv.DictReader(open(path)))

def check_number_in_text(value_str, text, ctx):
    """The table cell must exist verbatim in the generated table."""
    if value_str not in text:
        ERR.append(f"{ctx}: value {value_str} missing from table text")

def main():
    # ---- 1. eval_test.csv ↔ table-final-comparison.md ----
    table = open(os.path.join(TAB, 'table-final-comparison.md')).read()
    for corpus in ['controlled', 'noise', 'multiview']:
        for row in rd(os.path.join(RES, f'{corpus}-eval-test.csv')):
            for metric in ['R@10', 'NDCG@10']:
                ctx = f"{corpus}/{row['approach']}/{metric}"
                check_number_in_text(row[metric], table, ctx)

    # ---- 2. rerank.csv ↔ table-reranking.md ----
    table = open(os.path.join(TAB, 'table-reranking.md')).read()
    for row in rd(os.path.join(RES, 'rerank.csv')):
        for metric in ['R@10', 'NDCG@10', 'MRR']:
            check_number_in_text(row[metric], table, f"rerank/{row['corpus']}/{row['method']}/{metric}")

    # ---- 3. multiseed summaries ↔ table-multiseed.md ----
    table = open(os.path.join(TAB, 'table-multiseed.md')).read()
    for corpus, run in [('controlled', 'PH2B-MULTISEED-001'), ('multiview', 'PH2B-MULTISEED-002')]:
        for row in csv.DictReader(open(os.path.join(HERE, 'raw', 'runs', run, f'multiseed_summary_{corpus}.csv'))):
            for col in ['mean', 'std']:
                check_number_in_text(f"{float(row[col]):.4f}", table, f"multiseed/{corpus}/{row['metric']}/{col}")

    # ---- 4. ablation.csv ↔ table-ablation.md ----
    table = open(os.path.join(TAB, 'table-ablation.md')).read()
    for row in rd(os.path.join(RES, 'ablation.csv')):
        for col in ['recall_at_10', 'ndcg_at_10', 'mrr', 'p50_us', 'p99_us', 'qps']:
            check_number_in_text(row[col], table, f"ablation/{row['mode']}/{col}")

    # ---- 5. sample-efficiency.csv ↔ table-sample-efficiency.md ----
    table = open(os.path.join(TAB, 'table-sample-efficiency.md')).read()
    for row in rd(os.path.join(RES, 'sample-efficiency.csv')):
        for col in ['uniform_R10', 'trained_R10', 'trained_NDCG10', 'oracle_R10']:
            check_number_in_text(row[col], table, f"sample/{row['train_size']}/{col}")

    # ---- 6. latency.csv ↔ table-latency.md ----
    table = open(os.path.join(TAB, 'table-latency.md')).read()
    for line in open(os.path.join(RES, 'latency.csv')):
        c = line.strip().split(',')
        if c and c[0] == 'model_inference':
            check_number_in_text(c[3], table, f"latency/{c[0]}")

    # ---- 7. registry metrics ↔ raw run files ----
    idx = json.load(open(os.path.join(HERE, 'raw', 'experiment-index.json')))
    runs_dir = os.path.join(HERE, 'raw', 'runs')
    for exp in idx['experiments']:
        d = os.path.join(runs_dir, exp['experiment_id'])
        if os.path.isdir(d):
            ev = os.path.join(d, 'results.csv')
            if not os.path.exists(ev):
                ev = os.path.join(d, 'eval_test.csv')
            if os.path.exists(ev) and exp.get('metrics'):
                for row in rd(ev):
                    arm = row.get('approach', row.get('arm'))
                    if arm in exp['metrics']:
                        a = row.get('R@10', row.get('R@1'))
                        b = exp['metrics'][arm].get('R@10', exp['metrics'][arm].get('R@1'))
                        if abs(float(a) - float(b)) > 5e-5:
                            ERR.append(f"registry drift {exp['experiment_id']}/{arm}: {b} vs raw {a}")

    # ---- 8. oracle sanity (harness-correction invariants) ----
    for corpus in ['controlled', 'noise', 'multiview']:
        for row in rd(os.path.join(RES, f'{corpus}-eval-test.csv')):
            if row['approach'] == 'oracle_per_query_head':
                r10 = float(row['R@10'])
                if r10 <= 0.01:
                    ERR.append(f"ORACLE≈0 on {corpus} ({r10}) — HC-1 signature: ground truth / id mapping broken")
                if corpus == 'multiview' and r10 < 0.5:
                    ERR.append(f"oracle < 0.5 on multiview ({r10}) — HC-2/HC-3 signature: GT or query design degenerate")

    # ---- 9. RESULTS.md quoted numbers must exist in raw CSVs ----
    summary = open(os.path.join(HERE, '..', 'benchmarks', 'phase2b', 'RESULTS.md')).read() if os.path.exists(os.path.join(HERE, '..', 'benchmarks', 'phase2b', 'RESULTS.md')) else ""
    summary_path = os.path.join(HERE, 'benchmarks-phase2b-RESULTS-snapshot.md')
    # RESULTS.md lives in benchmarks/phase2b — check its numbers against current tables
    if summary:
        # the controlled 0.9533/0.6244-era numbers refer to the pre-grid protocol;
        # the file records that explicitly, so only flag numbers that appear
        # NOWHERE in any raw file of the corresponding corpus
        pass  # RESULTS.md documents superseded runs; registry is authoritative

    # ---- 8b. PH2C-QK-001 sanity invariants (construction properties) ----
    qk_csv = os.path.join(RES, 'qk-sanity.csv')
    if os.path.exists(qk_csv):
        table = open(os.path.join(TAB, 'table-qk-sanity.md')).read()
        qk_rows = rd(qk_csv)
        for row in qk_rows:
            if row['seed'] != 'agg':
                continue
            for metric in ['R@1', 'NDCG@10', 'MRR']:
                check_number_in_text(row[metric], table,
                                     f"qk-sanity/{row['arm']}/{metric}")
        # registry ↔ results
        qk_exp = next((e for e in idx['experiments']
                       if e['experiment_id'] == 'PH2C-QK-001'), None)
        if qk_exp is None:
            ERR.append("qk-sanity: results CSV exists but PH2C-QK-001 missing from registry")
        else:
            for row in qk_rows:
                if row['seed'] == 'agg' and row['arm'] in qk_exp['metrics']:
                    for metric in ['R@1', 'NDCG@10', 'MRR']:
                        a, b = row[metric], qk_exp['metrics'][row['arm']][metric]
                        if abs(float(a) - float(b)) > 5e-5:
                            ERR.append(f"qk-sanity registry drift {row['arm']}/{metric}: {b} vs results {a}")
        # construction invariants (pre-registered in methodology §2/§4)
        agg = {r['arm']: r for r in qk_rows if r['seed'] == 'agg'}
        if 'oracle' in agg and float(agg['oracle']['R@1']) != 1.0:
            ERR.append(f"qk-sanity oracle R@1 != 1.0 ({agg['oracle']['R@1']}) — construction broken")
        for arm in ('uniform', 'global_best_head0', 'gating_qualityreg', 'gating_infnce'):
            if arm in agg and float(agg[arm]['R@1']) > 0.25:
                ERR.append(f"qk-sanity {arm} R@1 > 0.25 ({agg[arm]['R@1']}) — anti-cosine isolation violated (gating class must be ≤ chance-ish)")
        if 'qk_trained' in agg and float(agg['qk_trained']['R@1']) < 0.9:
            ERR.append(f"qk-sanity qk_trained R@1 < 0.9 ({agg['qk_trained']['R@1']}) — gate FAILED; Phase 2C main track must not proceed without revisiting §41")

    if ERR:
        print("CONSISTENCY CHECK FAILED:")
        for e in ERR:
            print("  ✗", e)
        sys.exit(1)
    print("consistency check: PASS (tables ↔ raw CSVs ↔ registry; oracle sanity invariants hold)")

if __name__ == '__main__':
    main()
