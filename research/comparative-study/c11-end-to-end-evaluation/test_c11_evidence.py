import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from c11_analyze import main, paired_bootstrap, percentile, row_mrr
from c11_validate import validate

class C11EvidenceTests(unittest.TestCase):
    def test_frozen_run_plan_is_valid(self):
        self.assertEqual(validate(), [])

    def test_nearest_rank_percentile(self):
        self.assertEqual(percentile([5, 1, 3, 2, 4], .5), 3)
        self.assertEqual(percentile([5, 1, 3, 2, 4], .95), 5)

    def test_paired_bootstrap_is_deterministic_and_directional(self):
        control = {(1, 'q1'): {'ndcg10_qrels': .2}, (1, 'q2'): {'ndcg10_qrels': .4}, (1, 'q3'): {'ndcg10_qrels': .6}}
        candidate = {(1, 'q1'): {'ndcg10_qrels': .3}, (1, 'q2'): {'ndcg10_qrels': .5}, (1, 'q3'): {'ndcg10_qrels': .7}}
        a = paired_bootstrap(candidate, control, 'ndcg10_qrels', draws=1000, seed=42)
        b = paired_bootstrap(candidate, control, 'ndcg10_qrels', draws=1000, seed=42)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a['delta_mean'], .1, places=8)
        self.assertAlmostEqual(a['ci95_low'], .1, places=8)
        self.assertAlmostEqual(a['ci95_high'], .1, places=8)
        self.assertEqual(a['n_paired'], 3)

    def test_unpaired_metric_is_marked_unavailable(self):
        result = paired_bootstrap({(1,'q'): {'recall10_qrels': 1.0}}, {}, 'recall10_qrels')
        self.assertIsNone(result['ci95_low'])
        self.assertEqual(result['n_paired'], 0)

    def test_mrr_from_ranked_hits(self):
        self.assertEqual(row_mrr({'relevant_ids':['d2'], 'hits':['d1','d2']}), .5)
        self.assertEqual(row_mrr({'relevant_ids':['d2'], 'hits':['d1']}), 0.0)

    def test_analyzer_strict_mode_rejects_incomplete_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'raw'
            run = root / 'C11-SCI-A'
            artifacts = run / 'artifacts'
            artifacts.mkdir(parents=True)
            provenance = {'commit_sha':'a'*40, 'rustc_version':'rustc test', 'cpu_model':'test-cpu',
                          'logical_cpu_count':2, 'memory_total_bytes':1024, 'c8pilot_sha256':'b'*64,
                          'input_hashes':{'query':'c'*64}}
            (artifacts / 'environment.yaml').write_text(json.dumps(provenance), encoding='utf-8')
            row = {'query_id':'q1', 'recall10_qrels':1.0, 'ndcg10_qrels':1.0, 'latency_us':10,
                   'relevant_ids':['d1'], 'hits':['d1']}
            for rep in range(1, 6):
                (artifacts / f'ARM-A-rep{rep}.json').write_text(json.dumps({'per_query':[row]}), encoding='utf-8')
            metrics = {'run_id':'C11-SCI-A','mode':'A','dataset':'DS-SCIFACT','split':'TEST',
                       'rep_count':5,'test_queries':300,'candidate_budget':500,'ef_search':64,
                       'union_identity_ok':True,'per_rep':[{'recall10_qrels_mean':1.0,'ndcg10_qrels_mean':1.0,
                       'mrr10_qrels_mean':1.0,'p50_us':10,'p90_us':10,'p95_us':10,'deadline_exceeded':0} for _ in range(5)]}
            (run / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
            out = Path(tmp) / 'out'
            argv = ['c11_analyze.py','--raw-root',str(root),'--output-dir',str(out),'--require-complete']
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                status = main()
            result = json.loads((out / 'c11_results.json').read_text(encoding='utf-8'))
            self.assertEqual(status, 2)
            self.assertEqual(result['observed_cells'], 1)
            self.assertEqual(len(result['missing_cells']), 17)
            self.assertFalse(result['empirical_completion'])

if __name__ == '__main__': unittest.main()