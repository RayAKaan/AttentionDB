import unittest
from c11_analyze import paired_bootstrap, percentile, row_mrr
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

if __name__ == '__main__': unittest.main()