"""Validate shared banks, complete runs and seed-level suite aggregation."""
import copy
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import evaluate_fixed_context as evaluation
import evaluate_fixed_context_suite as suite
from test_fixed_context_cli import fixture_run


ROOT = Path(__file__).resolve().parents[1]


def suite_runs(tmp_path, monkeypatch, seeds=(0, 1)):
    log, _, _, _, variant = fixture_run(tmp_path, monkeypatch)
    log_root = tmp_path / 'logs'
    for seed in seeds:
        for experiment in suite.DEFAULT_EXPERIMENTS:
            target = log_root / 'toy/gentle' / ('seed%d' % seed) / experiment
            shutil.copytree(log, target)
            v = copy.deepcopy(variant)
            v['seed'] = seed
            (target / 'variant.json').write_text(json.dumps(v), encoding='utf-8')
    output = tmp_path / 'suite'
    args = ['--log-root', str(log_root), '--env-name', 'toy', '--seed-list', *map(str, seeds),
            '--output-root', str(output), '--epochs', '0', '2', '--collector-epochs', '0', '2',
            '--context-size', '2', '--context-repeats', '2', '--probe-size', '3', '--eval-seeds', '7', '8']
    return output, args


def test_complete_suite_uses_exact_bank_per_seed_and_statistics_over_training_seeds(tmp_path, monkeypatch):
    output, args = suite_runs(tmp_path, monkeypatch)
    import rlkit.torch.fixed_context_eval as core
    original = core.collect_zero_context
    calls = []

    def collect(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(core, 'collect_zero_context', collect)
    suite.main(args)
    assert len(calls) == 8  # two seeds, two collectors, one task, two repeats
    manifest = json.loads((output / 'suite_manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'complete' and len(manifest['completed']) == 6
    for seed in [0, 1]:
        results = [json.loads((output / ('seed%d' % seed) / experiment / 'fixed_context_results.json').read_text())
                   for experiment in suite.DEFAULT_EXPERIMENTS]
        assert len({r['bank_archive_sha256'] for r in results}) == 1
        assert all(Path(r['bank_dir']) == output / ('seed%d' % seed) / 'context_bank' for r in results)
        assert results[0]['records'] == results[1]['records'] == results[2]['records']
    with (output / 'per_seed_summary.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 36  # six runs, two checkpoints, three context sources
    with (output / 'group_summary.csv').open(newline='') as handle:
        groups = list(csv.DictReader(handle))
    assert len(groups) == 18
    assert all(int(row['training_seeds']) == 2 for row in groups)
    with (output / 'paired_summary.csv').open(newline='') as handle:
        pairs = list(csv.DictReader(handle))
    assert len(pairs) == 18
    assert all(float(row['mean_difference']) == 0 for row in pairs)
    # A repeated suite must reuse the archived bank instead of collecting again.
    monkeypatch.setattr(core, 'collect_zero_context', lambda *a, **kw: pytest.fail('must reuse bank'))
    suite.main(args)
    assert json.loads((output / 'suite_manifest.json').read_text())['status'] == 'complete'


def test_summary_and_paired_sd_use_training_seed_means(tmp_path):
    args = suite.parse_args(['--output-root', str(tmp_path), '--seed-list', '0', '1',
                             '--experiments', 'control', 'candidate'])
    rows = [dict(experiment=exp, seed=seed, epoch=499, source='offline', return_mean=value)
            for exp, vals in [('control', [10, 20]), ('candidate', [14, 18])]
            for seed, value in enumerate(vals)]
    suite.write_summaries(args, rows)
    with (tmp_path / 'group_summary.csv').open(newline='') as handle:
        groups = {row['experiment']: row for row in csv.DictReader(handle)}
    assert float(groups['control']['return_mean']) == 15
    assert float(groups['control']['training_seed_std']) == pytest.approx(math.sqrt(50))
    with (tmp_path / 'paired_summary.csv').open(newline='') as handle:
        pair = next(csv.DictReader(handle))
    assert float(pair['mean_difference']) == 1  # paired differences +4, -2
    assert float(pair['difference_seed_std']) == pytest.approx(math.sqrt(18))
    assert pair['positive_seeds'] == '1'
    assert json.loads(pair['differences_by_seed']) == {'0': 4, '1': -2}


def test_single_training_seed_has_no_estimated_between_seed_sd(tmp_path):
    args = suite.parse_args(['--output-root', str(tmp_path), '--seed-list', '0', '--experiments', 'one'])
    suite.write_summaries(args, [dict(experiment='one', seed=0, epoch=499, source='offline', return_mean=12)])
    with (tmp_path / 'group_summary.csv').open(newline='') as handle:
        assert next(csv.DictReader(handle))['training_seed_std'] == ''


@pytest.mark.parametrize('problem', ['missing', 'empty', 'seed', 'preprocessing'])
def test_every_run_is_checked_before_any_evaluation(tmp_path, monkeypatch, problem):
    output, args = suite_runs(tmp_path, monkeypatch, seeds=(0,))
    parsed = suite.parse_args(args)
    target = suite.run_dir(parsed, 0, suite.DEFAULT_EXPERIMENTS[-1])
    if problem in ('missing', 'empty'):
        path = target / 'policy_itr_2.pth'
        if problem == 'missing':
            path.unlink()
        else:
            path.write_bytes(b'')
    else:
        variant = json.loads((target / 'variant.json').read_text())
        if problem == 'seed':
            variant['seed'] = 2
        else:
            variant['algo_params']['max_path_length'] += 1
        (target / 'variant.json').write_text(json.dumps(variant))
    monkeypatch.setattr(evaluation, 'main', lambda *a: pytest.fail('must preflight all jobs'))
    with pytest.raises((FileNotFoundError, ValueError)):
        suite.main(args)
    assert not output.exists()


def test_dry_run_performs_no_evaluation_and_creates_no_output(tmp_path, monkeypatch):
    output, args = suite_runs(tmp_path, monkeypatch, seeds=(0,))
    monkeypatch.setattr(evaluation, 'main', lambda *a: pytest.fail('dry run must not evaluate'))
    suite.main(args + ['--dry-run'])
    assert not output.exists()


def test_failure_is_reported_and_no_partial_aggregate_is_published(tmp_path, monkeypatch):
    output, args = suite_runs(tmp_path, monkeypatch, seeds=(0,))

    def fail(*args):
        raise RuntimeError('evaluation failed')

    monkeypatch.setattr(evaluation, 'main', fail)
    with pytest.raises(RuntimeError, match='evaluation failed'):
        suite.main(args)
    manifest = json.loads((output / 'suite_manifest.json').read_text())
    assert manifest['status'] == 'failed' and manifest['completed'] == []
    assert not (output / 'group_summary.csv').exists()


def test_incomplete_or_unpaired_results_are_rejected(tmp_path, monkeypatch):
    output, argv = suite_runs(tmp_path, monkeypatch, seeds=(0,))
    args = suite.parse_args(argv)
    jobs = suite.preflight(args)
    for job in jobs[:2]:
        evaluation.main(suite.evaluation_args(args, job))
    references = {}
    suite.read_returns(args, jobs[0], references)
    path = jobs[1]['output_dir'] / 'fixed_context_results.json'
    result = json.loads(path.read_text())
    result['records'][0]['reset_seeds'][0] += 1
    evaluation.write_json(path, result)
    with pytest.raises(ValueError, match='reset seeds'):
        suite.read_returns(args, jobs[1], references)
    result['records'][0] = result['records'][1]
    evaluation.write_json(path, result)
    with pytest.raises(ValueError, match='Incomplete or duplicate'):
        suite.read_returns(args, jobs[1], references)


def test_help_does_not_require_training_runtime():
    source = "import sys, evaluate_fixed_context_suite as m; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; assert 'rlkit.envs' not in sys.modules; m.main(['--help'])"
    result = subprocess.run([sys.executable, '-c', source], cwd=str(ROOT), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--dry-run' in result.stdout


@pytest.mark.parametrize('options', [
    ['--experiments', '../bad'], ['--experiments', 'same', 'same'],
    ['--seed-list', '1', '1'], ['--seed-list', '-1'], ['--seed-list', str(2**32)],
    ['--eval-seeds', '-1'], ['--bank-seed', str(2**32)], ['--gpu', '-1'],
])
def test_invalid_suite_options(options):
    with pytest.raises(SystemExit):
        suite.parse_args(options)
