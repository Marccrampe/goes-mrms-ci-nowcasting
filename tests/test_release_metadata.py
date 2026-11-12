from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseMetadataTests(unittest.TestCase):
    def test_archived_test_table_and_source_hashes(self):
        folder = ROOT/'docs/results'
        for source in json.loads((folder/'sources.json').read_text()):
            self.assertEqual(hashlib.sha256((folder/source['package_file']).read_bytes()).hexdigest(), source['sha256'])
        with (folder/'main_TEST_results.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 5)
        for row in rows:
            tp, fp, fn = (int(row[k]) for k in ['TP', 'FP', 'FN'])
            self.assertEqual(int(row['n_issuances']), 4712)
            for key, value in [('CSI', tp/(tp+fp+fn)), ('POD', tp/(tp+fn)), ('FAR', fp/(tp+fp))]:
                self.assertAlmostEqual(float(row[key]), value, places=12)

    def test_catalogue_export_preserves_identity_and_rejects_ambiguous_splits(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            events = pd.DataFrame(dict(event_id=['E1', 'E2'], period_id=['P1', 'P2'], latitude=[41.123456789, 42.]))
            events.to_csv(d/'events.csv', index=False)
            split = pd.DataFrame(dict(period_id=['P1', 'P2'], split=['train', 'test']))
            split.to_csv(d/'splits.csv', index=False)
            cmd = [sys.executable, str(ROOT/'tools/export_catalogue_release.py'), '--events', str(d/'events.csv'),
                   '--period-splits', str(d/'splits.csv'), '--expected-events', '2']
            result = subprocess.run(cmd+['--output', str(d/'valid')], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            original = d/'valid/accepted_events_original.csv'
            self.assertEqual(original.read_bytes(), (d/'events.csv').read_bytes())
            got = pd.read_json(d/'valid/accepted_events.jsonl.gz', lines=True)
            self.assertEqual(list(got.event_id), ['E1', 'E2'])
            self.assertEqual(list(got.split), ['train', 'test'])
            split = pd.concat([split, pd.DataFrame(dict(period_id=['P1'], split=['test']))])
            split.to_csv(d/'splits.csv', index=False)
            result = subprocess.run(cmd+['--output', str(d/'invalid')], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((d/'invalid').exists())


if __name__ == '__main__':
    unittest.main()
