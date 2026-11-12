"""Saved-case regression and scientifically meaningful full-domain invariants."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from object_demo import CaseStudy
import object_demo_kernels as K


class CaseDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = CaseStudy(ROOT/'examples/fig11_case')

    def test_saved_counts_and_associations(self):
        # Constructor also checks exact original matched IDs and promoted candidate IDs.
        r = self.case.evaluate()
        self.assertEqual([r['counts'][k] for k in ('tp', 'fp', 'fn', 'accepted_growth')], [7, 10, 9, 8])
        np.testing.assert_allclose(r['metrics']['CSI / diagnostic CSI*'], [7/26, 15/26], atol=1e-12)

    def test_persistence_does_not_change_strict_objects_or_fn(self):
        ref = self.case.evaluate()
        for duration in (0, 5, 10, 15, 20):
            r = self.case.evaluate(persistence=duration)
            np.testing.assert_array_equal(r['predicted'], ref['predicted'])
            self.assertEqual(r['pairs'], ref['pairs'])
            self.assertEqual(r['counts']['fn'], 9)
        eligible5 = {e['id'] for e in self.case.candidates if e['persistence_flags'][5] is not None}
        eligible10 = {e['id'] for e in self.case.candidates if e['persistence_flags'][10] is not None}
        self.assertTrue(eligible10 <= eligible5)

    def test_protocol_invariants_across_controls(self):
        for q, duration, radius, area in [(q, d, r, a) for q in (.005, .05, .25)
                  for d, r, a in [(5, 5., 1), (10, 15., 2), (20, 30., 5)]]:
            r = self.case.evaluate(q, duration, radius, area); n = r['counts']
            self.assertEqual(n['tp'] + n['fn'], 16)
            self.assertEqual(n['tp'] + n['fp'], n['predicted_objects'])
            self.assertEqual(n['accepted_growth'] + n['remaining_FP'], n['fp'])
            self.assertFalse(set(r['accepted']) & r['hit_pred'])
            self.assertFalse(set(r['accepted']) & r['duplicates'])
            self.assertEqual(len({e['id'] for e in r['accepted'].values()}), n['accepted_growth'])
            sizes = np.bincount(r['predicted'].ravel())[1:]
            self.assertTrue(np.all(sizes >= 2))

    def test_empty_forecast_keeps_all_observed_fn(self):
        c = self.case
        r = K.classify(np.zeros_like(c.p), c.core, [], c.coverage, .05, 5, 15, c.dy, c.dx, K.C, K.N)
        self.assertEqual([r['counts'][k] for k in ('tp', 'fp', 'fn')], [0, 0, 16])

    def test_empty_truth_and_no_growth_never_promote_forecasts(self):
        c = self.case
        r = K.classify(c.p, np.zeros_like(c.core), [], c.coverage, .05, 5, 15, c.dy, c.dx, K.C, K.N)
        self.assertEqual(r['counts']['accepted_growth'], 0)
        self.assertEqual(r['counts']['fp'], 17)


if __name__ == '__main__':
    unittest.main()
