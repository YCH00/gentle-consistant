"""Local invariants only. These fixtures are never real model/environment smoke."""
import random
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from audit import PROTOCOL, write_json, sha256
from runtime_common import isolated_rng, seed32, context_hash, native_rms, normalize, tanh_gaussian_mean
from run_stage_a import stage_complete
from stage_a_selection import final_record


class RuntimeTests(unittest.TestCase):
    def test_rng_isolation_and_pairing(self):
        random.seed(73)
        np.random.seed(73)
        torch.manual_seed(73)
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        seed = seed32('v3', 'cheetah-vel', 0, 'evaluation_reset', repeat=1)
        with isolated_rng(seed):
            first = (random.random(), np.random.rand(), torch.rand(1))
        with isolated_rng(seed):
            second = (random.random(), np.random.rand(), torch.rand(1))
        self.assertEqual(first[:2], second[:2])
        self.assertTrue(torch.equal(first[2], second[2]))
        self.assertEqual(before[0], random.getstate())
        np.testing.assert_array_equal(before[1][1], np.random.get_state()[1])
        self.assertTrue(torch.equal(before[2], torch.get_rng_state()))
        self.assertNotEqual(seed, seed32('v3', 'cheetah-vel', 0, 'evaluation_reset', repeat=2))

    def test_hash_detects_row_and_dtype_changes(self):
        arr = np.arange(8, dtype=np.float32).reshape(4, 2)
        self.assertNotEqual(context_hash({'a': arr}), context_hash({'a': arr[::-1]}))
        self.assertNotEqual(context_hash({'a': arr}), context_hash({'a': arr.astype(np.float64)}))

    def test_native_single_update_matches_direct_moments(self):
        obs = np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=np.float32)
        actual = native_rms(obs)
        count = len(obs) + 1e-4
        mean = obs.mean(0).astype(np.float64) * len(obs) / count
        m2 = np.full(2, 1e-4, dtype=np.float64) + obs.var(0) * len(obs) + obs.mean(0).astype(np.float64)**2 * 1e-4 * len(obs) / count
        np.testing.assert_array_equal(actual['mean'], mean)
        np.testing.assert_array_equal(actual['var'], m2 / count)
        np.testing.assert_allclose(normalize(obs, actual), (obs - mean) / np.sqrt(m2 / count + 1e-8))

    def test_unseen_pool_never_enters_training_statistics(self):
        from stage_a_worker import reconstruct_training_stats
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            observed = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
            for task in (0, 1):
                folder = root / ('goal_idx' + str(task))
                folder.mkdir()
                data = np.empty((2, 4), dtype=object)
                for i in range(2):
                    data[i] = [observed[i] + task * 10000, np.zeros(1), 1., observed[i] + 1]
                path = folder / 'trj_evalsample0_step1000000.npy'
                np.save(path, data)
                rows.append(dict(task_id=task, path=path.relative_to(root).as_posix(), sha256=sha256(path),
                    rows=2, dimensions=dict(observation=2, action=1, reward=1, next_observation=2)))
            manifest = root / 'dataset_manifest.json'
            write_json(manifest, dict(files=rows, data_action_coordinates='normalized'))
            reference = native_rms(observed)
            np.savez(root / 'shared.npz', **reference)
            plan = dict(shared={'fixture': {'dataset_manifest.json': {'path': str(manifest)},
                'observation_normalizer.npz': {'path': str(root / 'shared.npz')},
                'task_payload': dict(train_tasks=[0], eval_tasks=[1], tasks=[{}, {}])}})
            entry = dict(environment='fixture', configuration=dict(n_train_tasks=1, n_eval_tasks=1,
                algo_params=dict(data_dir=str(root), n_trj=1, train_epoch=1000000)))
            out = root / 'output'
            out.mkdir()
            stats, receipt = reconstruct_training_stats(entry, plan, out)
            for key in reference:
                np.testing.assert_array_equal(stats[key], reference[key])
            self.assertEqual(len(receipt['native_loader_order']), 1)
            self.assertFalse(receipt['evaluation_data_used'])
            self.assertTrue((out / 'offline_pool_verification.json').is_file())

    def test_posterior_mean_is_expected_tanh(self):
        mean = torch.tensor([[0., 1., -1., 3.]])
        var = torch.tensor([[10., 3., 3., .1]])
        actual, error = tanh_gaussian_mean(mean, var)
        self.assertLessEqual(error, 1e-4)
        self.assertLess(abs(actual[0, 0].item()), 1e-6)
        self.assertAlmostEqual(actual[0, 1].item(), -actual[0, 2].item(), places=6)
        self.assertGreater(abs(actual[0, 1].item() - torch.tanh(mean[0, 1]).item()), .1)
        # Independent dense integration validates the numerical expectation.
        x = np.linspace(-10., 10., 40001)
        for i in range(4):
            expected = np.trapz(np.tanh(float(mean[0, i]) + np.sqrt(float(var[0, i])) * x)
                               * np.exp(-x*x/2) / np.sqrt(2*np.pi), x)
            self.assertAlmostEqual(actual[0, i].item(), expected, places=5)

    def test_gate_rejects_missing_seed_and_failed_collector(self):
        jobs = [dict(environment=e, method=m, seed=s, role='evaluation', passed=True)
                for e in PROTOCOL['environments'] for m in PROTOCOL['methods'] for s in PROTOCOL['training_seeds']]
        jobs += [dict(environment=e, role='collector200', passed=True) for e in PROTOCOL['environments']]
        self.assertTrue(stage_complete({'unavailable': []}, jobs, True))
        self.assertFalse(stage_complete({'unavailable': []}, jobs[1:], True))
        jobs[-1]['passed'] = False
        self.assertFalse(stage_complete({'unavailable': []}, jobs, True))
        self.assertFalse(stage_complete({'unavailable': []}, jobs, False))

    def test_final_marker_rejects_smoke_and_outside_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'seed0'
            run.mkdir()
            write_json(run / 'final.json', {'completed_native_plan': True, 'smoke': True, 'checkpoint': 'smoke.pt'})
            with self.assertRaises(ValueError):
                final_record(run, 'ER-TRL')
            (Path(directory) / 'outside.pt').write_bytes(b'fixture')
            write_json(run / 'final.json', {'completed_native_plan': True, 'checkpoint': '../outside.pt'})
            with self.assertRaises(ValueError):
                final_record(run, 'ER-TRL')


if __name__ == '__main__':
    unittest.main()
