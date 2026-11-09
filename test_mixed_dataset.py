"""CPU-only integration tests for mixed data; run python -m unittest test_mixed_dataset."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from rlkit.data_management import offline_dataset as ds
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
ROOT = Path(__file__).resolve().parent


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def definition(path, name, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class MixedDataTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.raw, self.old, self.art, self.oldart = [self.root / n for n in ('mixed', 'old', 'v3', 'v1')]
        self.art.mkdir(); self.oldart.mkdir()
        tasks = dict(protocol_id='shared-main-v1', environment='cheetah-dir', train_tasks=[0, 1],
                     eval_tasks=[0, 1], horizon=3, tasks=[{'direction': -1}, {'direction': 1}])
        for folder in (self.art, self.oldart):
            dump(folder / 'task_manifest.json', tasks)
        dimensions = dict(observation=2, action=1, reward=1, next_observation=2)
        files, oldfiles, rows, collections = [], [], {}, {}
        for task in (0, 1):
            for i, behavior in enumerate(('random', 'medium', 'expert')):
                data = np.empty((i + 1, 4), dtype=object)
                for step in range(i + 1):
                    data[step] = [np.array([task, step], dtype=float), np.array([0.2]), float(i), np.array([task, step+1.])]
                relative = 'goal_idx%d/trj_evalsample%d.npy' % (task, i)
                path = self.raw / relative
                path.parent.mkdir(parents=True, exist_ok=True); np.save(path, data)
                files.append(dict(path=relative, task_id=task, trajectory_index=i, behavior=behavior,
                                  splits=['train'], dimensions=dimensions, rows=i+1, sha256=ds.file_hash(path)))
                oldpath = self.old / ('goal_idx%d/trj_evalsample%d_step100.npy' % (task, i))
                oldpath.parent.mkdir(parents=True, exist_ok=True)
                data[:, 2] = 99.; np.save(oldpath, data)
                oldfiles.append(dict(path=oldpath.relative_to(self.old).as_posix(), task_id=task,
                                     dimensions=dimensions, rows=i+1, sha256=ds.file_hash(oldpath)))
            rows[str(task)] = 6
            collection = self.raw / ('goal_idx%d/collection.json' % task)
            dump(collection, {'task': task}); collections[str(task)] = ds.file_hash(collection)
        dump(self.oldart / 'dataset_manifest.json', dict(files=oldfiles))
        np.savez(self.art / 'observation_normalizer.npz', mean=[0., 0.], var=[1., 1.])
        self.manifest = dict(protocol_id='shared-native-v3', environment='cheetah-dir', data_action_coordinates='normalized',
                             observation_coordinates='raw', reward_coordinates='raw', terminal_mode='legacy_episode_end',
                             task_manifest_sha256=ds.file_hash(self.art / 'task_manifest.json'),
                             normalizer_sha256=ds.file_hash(self.art / 'observation_normalizer.npz'), files=files,
                             task_rows=rows, behavior_trajectory_quotas=dict(random=1, medium=1, expert=1))
        self.provenance = dict(protocol_id='shared-native-v3', environment='cheetah-dir', collection_sha256=collections,
                               action_low=[-1.], action_high=[1.],
                               old_dataset_manifest_sha256=ds.file_hash(self.oldart / 'dataset_manifest.json'))
        self.refresh()
        dump(self.art / 'validation.json', dict(artifact_status='passed', files=6, task_rows=rows))
        self.params = dict(train_data_format='mixed_rme_v3', data_dir=str(self.raw), train_artifacts_dir=str(self.art),
                           eval_data_dir=str(self.old), eval_artifacts_dir=str(self.oldart), n_trj=3,
                           train_epoch=999999, eval_epoch=100, max_path_length=3, use_next_obs_in_context=False)

    def refresh(self):
        dump(self.art / 'dataset_manifest.json', self.manifest)
        self.provenance['core_sha256'] = {n: ds.file_hash(self.art / n) for n in
                                        ('task_manifest.json', 'dataset_manifest.json', 'observation_normalizer.npz')}
        dump(self.art / 'preparation_provenance.json', self.provenance)

    def test_overlapping_ids(self):
        train = ds.training_paths(self.params, [0, 1]); evaluation = ds.evaluation_paths(self.params, [0, 1])
        self.assertEqual(len(train), 6); self.assertEqual(len(evaluation), 6)
        self.assertTrue(set(train).isdisjoint(evaluation))
        self.assertEqual(np.load(train[0], allow_pickle=True)[0, 2], 0)
        self.assertEqual(np.load(evaluation[0], allow_pickle=True)[0, 2], 99)

    def test_actual_main_buffer_boundaries_and_normalization(self):
        init = definition(ROOT / 'rlkit/core/rl_algorithm.py', 'init_buffer', dict(
            np=np, training_paths=ds.training_paths, evaluation_paths=ds.evaluation_paths, task_id=ds.task_id))
        Normalizer = definition(ROOT / 'rlkit/torch/pytorch_util.py', 'RunningMeanStd', dict(np=np))
        class Buffers:
            def __init__(self):
                self.task_buffers = {i: SimpleReplayBuffer(100, 2, 1, 1) for i in (0, 1)}
            def add_sample(self, task, obs, action, reward, terminal, next_obs, **kwargs):
                self.task_buffers[task].add_sample(obs, action, reward, terminal, next_obs, **kwargs)
        instance = SimpleNamespace(train_tasks=[0, 1], eval_tasks=[0, 1], load_eval_buffer=True,
            obs_normalizer=Normalizer(shape=2), train_buffer=Buffers(), eval_buffer=Buffers(),
            env=SimpleNamespace(update_obs_mean_var=lambda *args: None))
        init(instance, self.params)
        train, evaluation = instance.train_buffer.task_buffers[0], instance.eval_buffer.task_buffers[0]
        self.assertEqual(train._size, 6)
        np.testing.assert_array_equal(train._terminals[:6, 0], [1, 0, 1, 0, 0, 1])
        np.testing.assert_array_equal(train._rewards[:6, 0], [0, 1, 1, 2, 2, 2])
        np.testing.assert_array_equal(evaluation._rewards[:6, 0], np.full(6, 99.))
        np.testing.assert_allclose(train._observations[:6], evaluation._observations[:6])

    def test_cheetah_dir_evaluation_routes_to_historical_buffer(self):
        from collections import OrderedDict
        evaluate = definition(ROOT / 'rlkit/core/rl_algorithm.py', 'evaluate', dict(
            np=np, OrderedDict=OrderedDict, eval_util=SimpleNamespace(dprint=lambda *args: None),
            logger=SimpleNamespace(record_tabular=lambda *args: None)))
        calls = []
        def offline(indices, epoch, buffer):
            calls.append(buffer)
            return [1., 2.], [[1.], [2.]]
        instance = SimpleNamespace(eval_statistics=None, train_tasks=[0, 1], eval_tasks=[0, 1],
            separate_eval_source=True, train_buffer=object(), eval_buffer=object(),
            _do_eval=offline, _do_eval_with_online_context=lambda *args, **kwargs: ([1., 2.], [[1.], [2.]], None),
            agent=SimpleNamespace(log_diagnostics=lambda *args: None), render_eval_paths=False,
            plotter=None, num_iterations=10)
        evaluate(instance, 0)
        self.assertIs(calls[0], instance.eval_buffer)
        instance.separate_eval_source = False
        evaluate(instance, 0)
        self.assertIs(calls[1], instance.train_buffer)

    def test_runner_smoke_pipeline_is_separate(self):
        import shutil
        from unittest.mock import patch
        import run_mixed_gentle as runner
        repo = self.root / 'repo'
        shared = self.root / 'shared_protocol'
        shutil.copytree(self.raw, repo / 'data_mixed_rme_v3/cheetah-dir')
        shutil.copytree(self.old, repo / 'data/cheetah-dir')
        shutil.copytree(self.art, shared / 'shared-native-v3/cheetah-dir')
        shutil.copytree(self.oldart, shared / 'shared-main-v1/cheetah-dir')
        config = self.root / 'base.json'
        dump(config, dict(env_name='cheetah-dir', n_train_tasks=2, n_eval_tasks=2,
            env_params=dict(n_tasks=2), algo_params=self.params))
        args = ['runner', 'smoke', '--env', 'cheetah-dir', '--config', str(config)]
        with patch.object(runner, 'ROOT', repo), patch('sys.argv', args), patch.object(runner.subprocess, 'run') as run:
            runner.main()
        self.assertEqual(run.call_count, 3)
        stages = [Path(call.args[0][1]).name for call in run.call_args_list]
        self.assertEqual(stages, ['pretrain_dynamics.py', 'pretrain_encoder_decoder.py', 'train_gentle.py'])
        saved = ds.read_json(repo / 'output/mixed_rme_v3_smoke/cheetah-dir/config.json')
        self.assertEqual(saved['algo_params']['dataset_tag'], 'mixed_rme_v3_smoke')
        self.assertEqual(saved['dynamics_max_epochs'], 1)
        self.assertEqual(saved['algo_params']['num_iterations'], 2)
        self.assertNotIn('--exp_name', run.call_args.args[0])

    def test_corruption(self):
        with (self.raw / self.manifest['files'][0]['path']).open('ab') as stream: stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): ds.training_paths(self.params, [0, 1])

    def test_wrong_quota(self):
        self.manifest['behavior_trajectory_quotas']['random'] = 2; self.refresh()
        with self.assertRaisesRegex(ValueError, 'quota mismatch'): ds.training_paths(self.params, [0, 1])

    def test_wrong_rows(self):
        self.manifest['files'][0]['rows'] = 2; self.refresh()
        with self.assertRaisesRegex(ValueError, 'Row count mismatch'): ds.training_paths(self.params, [0, 1])

    def test_missing_file(self):
        (self.raw / self.manifest['files'][0]['path']).unlink()
        with self.assertRaises(FileNotFoundError): ds.training_paths(self.params, [0, 1])

    def test_legacy(self):
        self.assertEqual(len(ds.training_paths(dict(data_dir=str(self.old), train_epoch=100, n_trj=3), [0, 1])), 6)
        with self.assertRaisesRegex(ValueError, 'No legacy'): ds.legacy_paths(self.old, 200, 3, [0])

    def test_wrong_weight_identity(self):
        variant = dict(env_name='cheetah-dir', algo_params=self.params)
        directory = self.root / 'weights'; ds.write_identity(directory, variant); ds.check_identity(directory, variant)
        record = ds.read_json(directory / 'dataset_identity.json'); record['dataset_sha256'] = 'expert'
        dump(directory / 'dataset_identity.json', record)
        with self.assertRaisesRegex(ValueError, 'different dataset'): ds.check_identity(directory, variant)

    def test_environment_parameters(self):
        class Env:
            tasks = [{'direction': -1}, {'direction': 1}]
            action_space = SimpleNamespace(low=np.array([-1.]), high=np.array([1.]), shape=(1,))
            observation_space = SimpleNamespace(shape=(2,))
            _max_episode_steps = 3
            def get_all_task_idx(self): return range(2)
            def reset_task(self, task): pass
        env = Env(); env._wrapped_env = env
        variant = dict(env_name='cheetah-dir', algo_params=self.params, n_train_tasks=2, n_eval_tasks=2)
        ds.prepare_environment(env, variant)
        env.tasks = [{'direction': 0}, {'direction': 1}]
        with self.assertRaisesRegex(ValueError, 'task parameters'): ds.prepare_environment(env, variant)


if __name__ == '__main__': unittest.main()
