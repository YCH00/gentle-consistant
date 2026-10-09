"""Compare saved runs using one shared context bank per training seed.

Defaults compare experiment 37's no-consistency control with experiment 38's
two semantic arms. Evaluation runs sequentially and delegates the protocol,
checkpoint loading and context validation to evaluate_fixed_context.py.
"""
import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path

import evaluate_fixed_context as evaluation


DEFAULT_EXPERIMENTS = ['37_real_no_cons', '38_semantic_rl_only', '38_semantic_virtual_cons']


def path_component(value):
    if value in ('', '.', '..') or '/' in value or '\\' in value or ':' in value:
        raise argparse.ArgumentTypeError('must be a single directory name')
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-root', type=Path, default=Path('logs'))
    parser.add_argument('--env-name', type=path_component, default='ant-dir')
    parser.add_argument('--experiments', nargs='+', type=path_component, default=DEFAULT_EXPERIMENTS)
    parser.add_argument('--collector-exp', type=path_component, default='37_real_no_cons')
    parser.add_argument('--seed-list', '--seed_list', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--epochs', nargs='+', type=int, default=[200, 400, 499])
    parser.add_argument('--collector-epochs', nargs='+', type=int, default=[200, 499])
    parser.add_argument('--output-root', type=Path, default=Path('output/39_fixed_context/ant-dir'))
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--split', choices=['test', 'train'], default='test')
    parser.add_argument('--context-size', type=evaluation.positive_int, default=200)
    parser.add_argument('--context-repeats', type=evaluation.positive_int, default=3)
    parser.add_argument('--probe-size', type=evaluation.positive_int, default=256)
    parser.add_argument('--eval-seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--bank-seed', type=int, default=3600)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--offline-only', action='store_true')
    parser.add_argument('--save-trajectories', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Check every run/checkpoint and print the plan without evaluating.')
    args = parser.parse_args(argv)
    for name in ('seed_list', 'epochs', 'collector_epochs', 'eval_seeds'):
        values = getattr(args, name)
        if any(v < 0 for v in values) or len(set(values)) != len(values):
            parser.error('--' + name.replace('_', '-') + ' requires distinct nonnegative integers')
    if any(v >= 2**32 for v in args.seed_list + args.eval_seeds) or not 0 <= args.bank_seed < 2**32:
        parser.error('seed values must be below 2**32')
    if len(set(args.experiments)) != len(args.experiments):
        parser.error('--experiments must be distinct')
    if args.gpu is not None and args.gpu < 0:
        parser.error('--gpu must be nonnegative')
    args.epochs.sort()
    args.collector_epochs.sort()
    args.log_root = args.log_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.data_dir:
        args.data_dir = args.data_dir.expanduser().resolve()
    return args


def run_dir(args, seed, experiment):
    return args.log_root / args.env_name / 'gentle' / ('seed%d' % seed) / experiment


def preflight(args):
    """Reject a missing/incompatible run before any context is collected."""
    if args.output_root == args.log_root or args.log_root in args.output_root.parents:
        raise ValueError('Write suite outputs outside --log-root')
    jobs = []
    for seed in args.seed_list:
        collector_dir = run_dir(args, seed, args.collector_exp).resolve()
        collector = evaluation.load_variant(collector_dir)
        if collector['seed'] != seed or collector['env_name'] != args.env_name:
            raise ValueError('Collector variant does not match requested environment/training seed: ' + str(collector_dir))
        if args.data_dir:
            collector['algo_params']['data_dir'] = str(args.data_dir)
        if not args.offline_only:
            for epoch in args.collector_epochs:
                for path in evaluation.checkpoint_paths(collector_dir, epoch, ('policy',)).values():
                    if path.stat().st_size == 0:
                        raise ValueError('Empty collector checkpoint: ' + str(path))
        for experiment in args.experiments:
            log_dir = run_dir(args, seed, experiment).resolve()
            variant = evaluation.load_variant(log_dir)
            if variant['seed'] != seed or variant['env_name'] != args.env_name:
                raise ValueError('Run variant does not match requested environment/training seed: ' + str(log_dir))
            if args.data_dir:
                variant['algo_params']['data_dir'] = str(args.data_dir)
            evaluation.validate_collector_variant(variant, collector)
            for epoch in args.epochs:
                for path in evaluation.checkpoint_paths(log_dir, epoch).values():
                    if path.stat().st_size == 0:
                        raise ValueError('Empty evaluation checkpoint: ' + str(path))
            bank_dir = (args.output_root / ('seed%d' % seed) / 'context_bank').resolve()
            output_dir = (args.output_root / ('seed%d' % seed) / experiment).resolve()
            for destination in (args.output_root, bank_dir, output_dir):
                if any(destination == run or run in destination.parents for run in (log_dir, collector_dir)):
                    raise ValueError('Write suite outputs outside every training/collector run')
            jobs.append(dict(seed=seed, experiment=experiment, log_dir=log_dir,
                             collector_dir=collector_dir, bank_dir=bank_dir, output_dir=output_dir))
    return jobs


def evaluation_args(args, job):
    result = ['--log-dir', str(job['log_dir']), '--collector-log-dir', str(job['collector_dir']),
              '--bank-dir', str(job['bank_dir']), '--output-dir', str(job['output_dir']),
              '--epochs', *map(str, args.epochs), '--collector-epochs', *map(str, args.collector_epochs),
              '--context-size', str(args.context_size), '--context-repeats', str(args.context_repeats),
              '--probe-size', str(args.probe_size), '--eval-seeds', *map(str, args.eval_seeds),
              '--bank-seed', str(args.bank_seed), '--split', args.split]
    if args.data_dir:
        result += ['--data-dir', str(args.data_dir)]
    if args.gpu is not None:
        result += ['--gpu', str(args.gpu)]
    for flag in ('offline_only', 'save_trajectories'):
        if getattr(args, flag):
            result.append('--' + flag.replace('_', '-'))
    return result


def read_returns(args, job, references):
    path = job['output_dir'] / 'fixed_context_results.json'
    result = json.loads(path.read_text(encoding='utf-8'))
    if result['protocol'] != 'fixed-context-v1' or Path(result['log_dir']).resolve() != job['log_dir']:
        raise ValueError('Unexpected result protocol or run: ' + str(path))
    if Path(result['bank_dir']).resolve() != job['bank_dir']:
        raise ValueError('Evaluation did not use the shared context bank: ' + str(path))
    manifest = json.loads((job['bank_dir'] / 'manifest.json').read_text(encoding='utf-8'))
    if result['bank_archive_sha256'] != manifest['archive_sha256']:
        raise ValueError('Result and shared bank checksums differ: ' + str(path))
    expected = {(epoch, entry['source'], entry['task_id'], entry['repeat'])
                for epoch in args.epochs for entry in manifest['entries']}
    records = result['records']
    keys = [(r['epoch'], r['source'], r['task_id'], r['repeat']) for r in records]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Incomplete or duplicate evaluation records: ' + str(path))
    resets = {key: r['reset_seeds'] for key, r in zip(keys, records)}
    signature = (result['bank_archive_sha256'], resets)
    if job['seed'] in references and signature != references[job['seed']]:
        raise ValueError('Runs must share exact contexts and rollout reset seeds within a training seed')
    references[job['seed']] = signature
    groups = {}
    for r in records:
        value = float(r['evaluation']['return_mean'])
        if not math.isfinite(value):
            raise ValueError('Nonfinite evaluation return: ' + str(path))
        groups.setdefault((r['epoch'], r['source']), []).append(value)
    return [dict(experiment=job['experiment'], seed=job['seed'], epoch=epoch, source=source,
                 return_mean=statistics.mean(values)) for (epoch, source), values in sorted(groups.items())]


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summaries(args, rows):
    write_csv(args.output_root / 'per_seed_summary.csv', ['experiment', 'seed', 'epoch', 'source', 'return_mean'], rows)
    groups = {}
    values = {}
    for row in rows:
        key = (row['experiment'], row['epoch'], row['source'])
        groups.setdefault(key, []).append(row['return_mean'])
        values[(row['experiment'], row['seed'], row['epoch'], row['source'])] = row['return_mean']
    summary = [dict(experiment=exp, epoch=epoch, source=source, training_seeds=len(v),
                    return_mean=statistics.mean(v), training_seed_std=statistics.stdev(v) if len(v) > 1 else '')
               for (exp, epoch, source), v in sorted(groups.items())]
    write_csv(args.output_root / 'group_summary.csv',
              ['experiment', 'epoch', 'source', 'training_seeds', 'return_mean', 'training_seed_std'], summary)
    pairs = []
    for control, experiment in itertools.combinations(args.experiments, 2):
        for epoch, source in sorted({(r['epoch'], r['source']) for r in rows}):
            differences = [values[(experiment, seed, epoch, source)] - values[(control, seed, epoch, source)]
                           for seed in args.seed_list]
            pairs.append(dict(experiment=experiment, control=control, epoch=epoch, source=source,
                training_seeds=len(differences), mean_difference=statistics.mean(differences),
                difference_seed_std=statistics.stdev(differences) if len(differences) > 1 else '',
                positive_seeds=sum(v > 0 for v in differences),
                differences_by_seed=json.dumps(dict(zip(args.seed_list, differences)), allow_nan=False)))
    write_csv(args.output_root / 'paired_summary.csv',
              ['experiment', 'control', 'epoch', 'source', 'training_seeds', 'mean_difference',
               'difference_seed_std', 'positive_seeds', 'differences_by_seed'], pairs)


def main(argv=None):
    args = parse_args(argv)
    jobs = preflight(args)
    for job in jobs:
        print('seed%d %s: evaluate %s; collector=%s; shared bank=%s' % (
            job['seed'], job['experiment'], args.epochs, args.collector_exp, job['bank_dir']), flush=True)
    if args.dry_run:
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = dict(protocol='fixed-context-suite-v1', status='running', settings=vars(args), jobs=jobs,
                    completed=[], notes=[
                        'One exact context bank and collector per training seed, shared across experiments.',
                        'Returns average equally over tasks and context repeats within each training seed.',
                        'Group standard deviations and paired differences use training seeds, not evaluation resets.',
                        'Blank standard deviation means only one training seed was evaluated.',
                        'Only status=complete certifies that all requested runs and aggregate CSVs finished.',
                        'Fixed-context returns use a different evaluation protocol from historical _expl logs.'])
    manifest_path = args.output_root / 'suite_manifest.json'
    evaluation.write_json(manifest_path, manifest)
    rows, references = [], {}
    try:
        for job in jobs:
            evaluation.main(evaluation_args(args, job))
            rows.extend(read_returns(args, job, references))
            manifest['completed'].append(dict(seed=job['seed'], experiment=job['experiment']))
            evaluation.write_json(manifest_path, manifest)
        write_summaries(args, rows)
    except Exception as error:
        manifest.update(status='failed', error=str(error))
        evaluation.write_json(manifest_path, manifest)
        raise
    manifest['status'] = 'complete'
    evaluation.write_json(manifest_path, manifest)
    print('Completed %d evaluations. Summaries: %s' % (len(jobs), args.output_root), flush=True)


if __name__ == '__main__':
    main()
