import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from rlkit.data_management.offline_sources import (
    sha256, trajectory_paths, load_trajectory, validate_bundle,
)


class MixedDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.params = dict(data_dir=str(self.root / 'mixed'), eval_data_dir=str(self.root / 'old'),
                           mixed_artifacts_dir=str(self.root / 'mixed_art'), eval_artifacts_dir=str(self.root / 'old_art'))
        for mixed in (True, False):
            raw = Path(self.params['data_dir' if mixed else 'eval_data_dir'])
            art = Path(self.params['mixed_artifacts_dir' if mixed else 'eval_artifacts_dir'])
            art.mkdir()
            (raw / 'goal_idx0').mkdir(parents=True)
            entries = []
            for index, length in enumerate((2, 1)):
                array = np.empty((length, 4), dtype=object)
                for row in range(length):
                    array[row] = [np.array([100. if mixed else -100.]), np.array([.2 if mixed else .02]), 1., np.array([3.])]
                relative = 'goal_idx0/trj_evalsample%d%s.npy' % (index, '' if mixed else '_step100000')
                np.save(raw / relative, array)
                entries.append(dict(path=relative, task_id=0, trajectory_index=index, rows=length,
                    sha256=sha256(raw / relative), splits=['train'] if mixed else ['train', 'test'],
                    behavior='random', checkpoint=None, checkpoint_sha256=None, seed=index,
                    dimensions=dict(observation=1, action=1, reward=1, next_observation=1)))
            self.write(art / 'task_manifest.json', dict(train_tasks=[0], eval_tasks=[0], horizon=20))
            np.savez(art / 'observation_normalizer.npz', mean=[0], var=[1])
            manifest = dict(protocol_id='shared-native-v3' if mixed else 'shared-main-v1', environment='fixture',
                            files=entries, task_rows={'0': 3}, terminal_mode='legacy_episode_end',
                            data_action_coordinates='normalized' if mixed else 'physical',
                            task_manifest_sha256=sha256(art / 'task_manifest.json'),
                            normalizer_sha256=sha256(art / 'observation_normalizer.npz'))
            self.write(art / 'dataset_manifest.json', manifest)
            collection = raw / 'goal_idx0/collection.json'
            self.write(collection, {'files': [dict(e, path='fixture/' + e['path']) for e in entries]})
            self.write(art / 'preparation_provenance.json', dict(
                core_sha256={name: sha256(art / name) for name in ('task_manifest.json', 'dataset_manifest.json', 'observation_normalizer.npz')},
                collection_sha256={'0': sha256(collection)}, action_low=[-.1], action_high=[.1]))
            self.write(art / 'validation.json', dict(artifact_status='passed', files=2, task_rows={'0': 3}))

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding='utf-8')

    def test_sources_and_action_coordinates(self):
        train, _ = trajectory_paths(self.params, [0])
        evaluation, _ = trajectory_paths(self.params, [0], 'eval')
        self.assertEqual(load_trajectory(train[0], self.params)[0, 0][0], 100)
        old = load_trajectory(evaluation[0], self.params, 'eval')
        self.assertEqual(old[0, 0][0], -100)
        self.assertAlmostEqual(old[0, 1][0], .2)

    def test_corrupt_trajectory_rejected(self):
        path = Path(self.params['data_dir']) / 'goal_idx0/trj_evalsample0.npy'
        with path.open('ab') as output:
            output.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            trajectory_paths(self.params, [0])

    def test_legacy_filename_selection(self):
        params = dict(data_dir=self.params['eval_data_dir'], n_trj=2, train_epoch=100000, eval_epoch=None)
        paths, tasks = trajectory_paths(params, [0])
        self.assertEqual(tasks, [0, 0])
        self.assertTrue(all('_step100000.npy' in p for p in paths))

    def test_actual_buffer_loader_preserves_short_boundaries(self):
        # Execute the actual method without importing the MuJoCo training stack.
        source = Path('rlkit/core/rl_algorithm.py').read_text(encoding='utf-8')
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'init_buffer')
        namespace = dict(np=np, trajectory_paths=trajectory_paths, load_trajectory=load_trajectory)
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<buffer>', 'exec'), namespace)
        class Buffer:
            def __init__(self): self.rows = []
            def add_sample(self, *args, **kwargs): self.rows.append(args)
        normalizer = SimpleNamespace(update=lambda x: None, forward=lambda x: x, mean=0, var=1)
        model = SimpleNamespace(train_tasks=[0], eval_tasks=[0], load_eval_buffer=True,
            obs_normalizer=normalizer, env=SimpleNamespace(update_obs_mean_var=lambda *x: None),
            train_buffer=Buffer(), eval_buffer=Buffer())
        namespace['init_buffer'](model, self.params)
        self.assertEqual([row[4] for row in model.train_buffer.rows], [0, 1, 1])
        self.assertEqual([row[4] for row in model.eval_buffer.rows], [0, 1, 1])
        self.assertEqual(model.train_buffer.rows[0][1][0], 100)
        self.assertEqual(model.eval_buffer.rows[0][1][0], -100)


if __name__ == '__main__':
    unittest.main()
