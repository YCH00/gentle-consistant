"""Optional tests against downloaded source. RANDOM FIXTURE WEIGHTS; NOT stage A."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

EVIDENCE = Path(__file__).resolve().parent.parent / 'diagnostics_audits/evidence-expanded-20261102/files'
SOURCES = {'GENTLE': 'GENTLE-consistant', 'FOCAL': 'FOCAL-v2', 'CORRO': 'CORRO-native-v2',
           'CSRO': 'backup/CSRO', 'UNICORN-SS': 'UNICORN-ss-v2', 'ER-TRL': 'ER-TRL-v2'}


def fixture_case(method, environment):
    import numpy as np
    import torch
    from audit import PROTOCOL, sha256
    from native_adapters import NativeAdapter, native_source_digest
    from runtime_common import digest, isolated_rng, weights_hash
    torch.set_num_threads(1)
    root = EVIDENCE / SOURCES[method]
    sys.path.insert(0, str(root))
    if method == 'GENTLE':
        cfg = json.loads((root / 'logs' / environment / 'gentle/seed0/0_gentle_base/variant.json').read_text())
    else:
        path = next(p for p in root.rglob('resolved_config.json') if environment in p.parts and p.parent.name == 'seed0')
        cfg = json.loads(path.read_text())
    od, ad = (20 if environment == 'cheetah-vel' else 17), 6
    shared = {name: {'sha256': 'fixture-' + name} for name in
              ('task_manifest.json', 'dataset_manifest.json', 'observation_normalizer.npz')}
    hashes = {k: v['sha256'] for k, v in shared.items()}
    if method in ('GENTLE', 'FOCAL', 'CSRO', 'ER-TRL'):
        native = cfg if method in ('GENTLE', 'CSRO') else cfg['native']
        a, latent = native['algo_params'], native['latent_size']
        enc_module = 'rlkit.torch.autoencoder' if method == 'GENTLE' else 'rlkit.torch.networks'
        pol_module = 'rlkit.torch.agents.policies' if method == 'ER-TRL' else 'rlkit.torch.sac.policies'
        hidden = [200]*3 if method == 'FOCAL' else native['encoder_dims'] if method == 'ER-TRL' else [native['net_size']]*3
        kwargs = {'layer_norm': a.get('layer_norm', False)} if method == 'CSRO' else {}
        encoder = importlib.import_module(enc_module).MlpEncoder(hidden_sizes=hidden,
            input_size=od+ad+1+(od if a['use_next_obs_in_context'] else 0), output_size=latent, output_activation=torch.tanh, **kwargs)
        policy = importlib.import_module(pol_module).TanhGaussianPolicy(hidden_sizes=[native['net_size']]*3,
                   obs_dim=od+latent, latent_dim=latent, action_dim=ad)
        if method == 'GENTLE':
            states = [encoder.state_dict(), policy.state_dict()]
        elif method == 'CSRO':
            states = [dict(method=method, test_only=False, training_seed=0, iteration=499,
                scientific_config=importlib.import_module('csro_shared.configuration').scientific_config(cfg),
                artifact_hashes=hashes, source_fingerprint=native_source_digest(method, root),
                main_updates=500*a['num_train_steps_per_itr'],
                networks=dict(context_encoder=encoder.state_dict(), policy=policy.state_dict()))]
        else:
            native_plan = importlib.import_module('focal_v2.config' if method == 'FOCAL' else 'ertrl_v2.config').training_plan(cfg)
            states = [dict(binding=dict(config_hash=digest(cfg), training_seed=0, source_artifact_hashes=hashes,
                implementation_hashes={}, training_plan_hash=digest(native_plan)), smoke=False, completed_native_plan=True,
                main_updates=native_plan['main_updates'], meta_updates=a['num_iterations']*a['num_train_steps_per_itr'],
                optimizer_counts=native_plan['optimizer_steps' if method == 'FOCAL' else 'expected_optimizer_steps'],
                next_iteration=PROTOCOL['expected_final_iteration_zero_based'][method]+1,
                encoder=encoder.state_dict(), policy=policy.state_dict())]
    else:
        unicorn = method == 'UNICORN-SS'
        a = cfg['native'] if unicorn else cfg
        latent = a['task_embedding_size']
        kwargs = {'stochasity': True} if unicorn else {'normalize': a['normalize_z']}
        encoder = importlib.import_module('models.encoder').MLPEncoder(hidden_size=a['aggregator_hidden_size'],
            num_hidden_layers=2, task_embedding_size=latent, action_size=ad, state_size=od,
            reward_size=1, term_size=int(unicorn), **kwargs)
        policy = importlib.import_module('models.policy').TanhGaussianPolicy(obs_dim=od+latent,
                    action_dim=ad, hidden_sizes=a['policy_layers'])
        parts = dict(encoder=encoder.state_dict(), agent={'policy.'+k: v for k,v in policy.state_dict().items()})
        identity = dict(config_hash=digest(cfg), source_artifact_hashes=hashes, training_seed=0)
        identity['source_hash' if unicorn else 'implementation_hash'] = native_source_digest(method, root)
        if unicorn:
            states = [dict(identity=identity, iteration=1000,
                           model=dict(main_updates=a['num_iters']*a['rl_updates_per_iter'], modules=parts))]
        else:
            parts['context'] = importlib.import_module('models.encoder').SelfAttnEncoder(input_dim=latent, num_output_mlp=0).state_dict()
            states = [dict(identity=identity, progress=dict(stage='complete', policy_updates=a['num_iters']*a['rl_updates_per_iter']), modules=parts)]
    with tempfile.TemporaryDirectory() as directory:
        paths = []
        for idx, state in enumerate(states):
            path = Path(directory) / ('fixture-{}.pt'.format(idx))
            torch.save(state, path)
            paths.append(dict(path=str(path), sha256=sha256(path)))
        entry = dict(method=method, environment=environment, configuration=cfg, training_seed=0,
                     final_iteration_zero_based=PROTOCOL['expected_final_iteration_zero_based'][method],
                     source_root=str(root), checkpoint_files=paths)
        stats = dict(mean=np.ones(od), var=np.ones(od)*2, count=np.asarray(10.)) if method in ('GENTLE', 'CSRO') else None
        adapter = NativeAdapter(entry, shared, od, ad, stats=stats)
        frozen = weights_hash(adapter.modules)
        rng = np.random.default_rng(17)
        raw = dict(observations=rng.normal(size=(200,od)), next_observations=rng.normal(size=(200,od)),
                   actions=rng.uniform(-1,1,(200,ad)), rewards=rng.normal(size=(200,1)), episode_end=np.zeros((200,1)))
        with isolated_rng(1):
            rep = adapter.infer(raw, representation=True)
            cond = adapter.infer(raw)
        with isolated_rng(2):
            rep2 = adapter.infer(raw, representation=True)
        assert torch.equal(rep,rep2) and rep.shape == (1,latent)
        assert adapter.act(np.zeros(od),cond).shape == (ad,)
        assert frozen == weights_hash(adapter.modules)
        # Strict loading and checkpoint-integrity gate must reject changed weights.
        paths[0]['sha256'] = 'bad'
        try:
            NativeAdapter(entry,shared,od,ad,stats=stats)
        except ValueError:
            pass
        else:
            raise AssertionError('Corrupt checkpoint hash accepted')
    print('PASS source interface fixture',method,environment)


@unittest.skipUnless(EVIDENCE.is_dir(), 'Optional downloaded-source fixtures are not present')
class NativeInterfaceTests(unittest.TestCase):
    def test_native_interfaces_in_separate_processes(self):
        for method in SOURCES:
            for environment in ('cheetah-vel','walker-rand-params'):
                with self.subTest(method=method, environment=environment):
                    result = subprocess.run([sys.executable,str(Path(__file__).resolve()),'--case',method,environment],
                                            capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--case':
        fixture_case(sys.argv[2],sys.argv[3])
    else:
        unittest.main()
