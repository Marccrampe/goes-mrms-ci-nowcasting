import importlib.util
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ci_nowcasting.paths import activate_sources, resolve_path, project_root
activate_sources()

class ReleaseTests(unittest.TestCase):

    def test_real_checkpoint_and_miniature(self):
        from models.inference import load_model, physical_inputs, predict, sha256
        path = ROOT / 'weights/goes_mrms_ci35_hard_r2_public.pt'
        if not path.exists():
            self.fail('Download the released checkpoint before running release validation')
        self.assertEqual(sha256(path), '9bc69822fb18d9b5036bc4e90fa7487aa08aafdaeda364ebb176fc55b01d5b0b')
        model = load_model(path, 'cpu')
        x = physical_inputs(ROOT / 'examples/mini_case.npz', ROOT / 'preprocessing/dataset/normalization.json')
        with torch.no_grad():
            y = predict(model, x)
        self.assertEqual(tuple(y.shape), (1, 2, 64, 64))
        self.assertTrue(bool(torch.isfinite(y).all()))
        self.assertTrue(bool(((y >= 0) & (y <= 1)).all()))

    def test_archived_target_transform(self):
        from ci_target_inflation import numerical_selftest
        numerical_selftest()

    def test_original_matching_and_recovery_cases(self):
        import ci_common_exploratory_core as C
        from verification.strict_ci35 import competition
        sys.modules['competition'] = competition
        spec = importlib.util.spec_from_file_location('release_cases', ROOT / 'tests/original_competition_cases.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(len(mod.run_tests(C)), 15)

    def test_workspace_relocation(self):
        path = resolve_path('/project/data/example.npz')
        self.assertEqual(Path(path), project_root() / 'data/example.npz')
        source = resolve_path('/project/scripts/isolated_ci_binary_v1/core.py')
        self.assertTrue(Path(source).is_file())

    def test_two_model_input_contracts(self):
        from models.resunet import build_final_model, build_goes_only_model
        for build, n in [(build_final_model, 9), (build_goes_only_model, 7)]:
            model = build(width=8).eval()
            with torch.no_grad():
                y = model(torch.zeros(1, 4, n, 32, 32))
            self.assertEqual(tuple(y.shape), (1, 2, 32, 32))

    def test_public_model_matches_original_weights_and_logits(self):
        import train_ci_a_radar_features as archived
        from models.resunet import build_final_model
        a = archived.V3Model('resunet', True, 8, 'raw', 'both').eval()
        b = build_final_model(width=8).eval()
        b.load_state_dict(a.state_dict(), strict=True)
        x = torch.rand(1, 4, 9, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)

    def test_target_wrapper_preserves_validation_and_anchors(self):
        from unittest.mock import patch
        from preprocessing.dataset.dataset import TargetDataset, CIGoesV4Dataset
        y = np.zeros((2, 16, 16), np.float32)
        y[0, 7, 7] = 0.4
        mask = np.ones_like(y)
        anchors = [(7, 7, 0)]
        dataset = object.__new__(TargetDataset)
        dataset.training_kind = 'hard'
        dataset.extra_pixels = 2
        dataset.dy_km = dataset.dx_km = 2.0
        with patch.object(CIGoesV4Dataset, '_build_targets', return_value=(y, mask, anchors)):
            dataset.split = 'validation'
            v, m, a = dataset._build_targets(None)
            self.assertIs(v, y)
            self.assertIs(a, anchors)
            self.assertIs(m, mask)
            dataset.split = 'train'
            v, m, a = dataset._build_targets(None)
            self.assertTrue(np.isin(v, [0, 1]).all())
            self.assertGreater(v.sum(), 1)
            self.assertIs(a, anchors)
            self.assertIs(m, mask)

    def test_cpu_training_gradient_step(self):
        from models.resunet import build_final_model
        from train_ci_a_radar_features import ci_loss
        model = build_final_model(width=8).train()
        opt = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.0001)
        x = torch.rand(1, 4, 9, 32, 32)
        y = torch.zeros(1, 2, 32, 32)
        mask = torch.ones_like(y)
        y[..., 16, 16] = 1
        before = model.head.weight.detach().clone()
        loss, _, _ = ci_loss(model(x), y, mask, 1.0, 0.0, 1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertFalse(torch.equal(before, model.head.weight.detach()))

    def test_frozen_threshold_table(self):
        c = json.loads((ROOT / 'configs/frozen_thresholds.json').read_text())
        self.assertEqual(c['selection_split'], 'validation')
        self.assertEqual(c['thresholds']['I4_CI35'], [0.075, 0.015])
        self.assertEqual(len(c['thresholds']), 1)

    def test_strict_cli_counts_and_separate_association(self):
        import tempfile, subprocess, csv
        with tempfile.TemporaryDirectory(prefix='ci verification ') as d:
            temp = Path(d)
            core = np.zeros((1, 2, 28, 32), bool)
            core[:, :, 5, 5] = True
            core[:, :, 5, 15] = True
            prob = np.zeros(core.shape, np.float32)
            prob[:, :, 5, 5:16] = 0.8
            prob[:, :, 22, 23:26] = 0.8
            prob[:, :, 18, 2] = 0.8
            np.savez_compressed(temp / 'input.npz', probability=prob, core=core, issuance_ns=np.array([1], np.int64), period_id=np.array(['synthetic']), dy_km=np.array(2.0), dx_km=np.array(2.0), split=np.array('test'))
            subprocess.run([sys.executable, str(ROOT / 'verification/strict_ci35/score.py'), '--input', str(temp / 'input.npz'), '--thresholds', str(ROOT / 'configs/frozen_thresholds.json'), '--model', 'I4_CI35', '--output', str(temp / 'results'), '--association'], check=True, cwd=temp, capture_output=True, text=True)
            with (temp / 'results/strict_scores.csv').open() as f:
                rows = list(csv.DictReader(f))
            with (temp / 'results/association_scores.csv').open() as f:
                assoc = list(csv.DictReader(f))
            for row, new in zip(rows, assoc):
                self.assertEqual([int(row[k]) for k in ['TP', 'FP', 'FN']], [1, 1, 1])
                self.assertEqual(int(new['FN_recovered']), 1)
                self.assertEqual(int(new['FP_new']), 1)
                self.assertEqual(int(new['FN_new']), 0)
if __name__ == '__main__':
    unittest.main()
