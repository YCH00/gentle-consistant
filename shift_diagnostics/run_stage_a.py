"""Run source verification, fixed final-model selection and isolated real smoke jobs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

from audit import PROTOCOL, sha256, write_json
from stage_a_selection import select_models


def verify_inputs(plan):
    if plan['protocol'] != PROTOCOL:
        raise ValueError('Frozen plan differs from protocol.json; regenerate the plan from verified evidence')
    records = list(plan['source_files'])
    for shared in plan['shared'].values():
        records += [value for value in shared.values() if isinstance(value, dict) and 'path' in value and 'sha256' in value]
    errors = []
    for item in records:
        path = Path(item['path'])
        if not path.is_file() or sha256(path) != item['sha256']:
            errors.append(str(path))
    if errors:
        raise ValueError('Evidence-pinned inputs changed/missing: ' + repr(errors[:20]))
    return {'verified_files': len(records), 'evidence_sha256': plan['evidence_sha256']}


def worker_environment(environment, explicit):
    """Set the dynamic linker search path BEFORE starting a Python interpreter."""
    version = 131 if environment == 'walker-rand-params' else 150
    env = os.environ.copy()
    choices = [explicit, env.get('MUJOCO_PY_MJPRO_PATH'), str(Path.home() / '.mujoco' / ('mjpro' + str(version)))]
    selected = next((Path(p).expanduser().resolve() for p in choices if p and Path(p).expanduser().is_dir()
                     and Path(p).name == 'mjpro' + str(version)), None)
    if selected is None:
        raise RuntimeError('Existing mjpro{} directory missing; use --mujoco{}'.format(version, version))
    env['MUJOCO_PY_MJPRO_PATH'] = str(selected)
    # Remove the other version's bin directory inherited from a training shell.
    previous = [p for p in env.get('LD_LIBRARY_PATH', '').split(os.pathsep)
                if p and 'mjpro131' not in p and 'mjpro150' not in p]
    env['LD_LIBRARY_PATH'] = os.pathsep.join([str(selected / 'bin')] + previous)
    env['PYTHONUNBUFFERED'] = '1'
    return env


def stage_complete(selection, jobs, inputs_unchanged):
    expected = {(env, method, seed) for env in PROTOCOL['environments']
                for method in PROTOCOL['methods'] for seed in PROTOCOL['training_seeds']}
    passed = {(j['environment'], j['method'], j['seed']) for j in jobs
              if j.get('role') == 'evaluation' and j.get('passed')}
    collectors = {j['environment'] for j in jobs if j.get('role') == 'collector200' and j.get('passed')}
    return bool(inputs_unchanged and not selection['unavailable'] and passed == expected
                and collectors == set(PROTOCOL['environments']))


def run(args, out):
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    print('Verifying evidence-pinned sources and shared artifacts...', flush=True)
    checked = verify_inputs(plan)
    diagnostic_sources = {str(p.resolve()): sha256(p) for p in Path(__file__).parent.iterdir()
                          if p.suffix in ('.py', '.json') and p.is_file()}
    write_json(out / 'diagnostic_source_hashes.json', diagnostic_sources)
    write_json(out / 'input_verification.json', checked)
    write_json(out / 'stage_a_plan.json', plan)
    write_json(out / 'protocol.json', PROTOCOL)
    print('Selecting final models before any diagnostic rollout (large checkpoints are hashed)...', flush=True)
    selection = select_models(plan)
    write_json(out / 'model_selection.json', selection)
    write_json(out / 'models.json', selection)
    print('Selected {} evaluation models and {} iteration-200 collectors; unavailable records={}'.format(
          len(selection['models']), len(selection['collectors']), len(selection['unavailable'])), flush=True)
    jobs = []
    (out / 'logs').mkdir()
    script = Path(__file__).with_name('stage_a_worker.py').resolve()

    def launch(environment, role, destination, method=None, seed=0, bank=None):
        job = dict(environment=environment, role=role, method=method, seed=seed, output=str(destination), passed=False)
        cmd = [sys.executable, str(script), '--plan', str(out / 'stage_a_plan.json'),
               '--selection', str(out / 'model_selection.json'), '--environment', environment,
               '--out', str(destination), '--device', args.device]
        if role == 'bank':
            cmd += ['--prepare']
        else:
            cmd += ['--method', method, '--seed', str(seed), '--bank', str(bank)]
            if role == 'collector200':
                cmd += ['--collector200']
        log = out / 'logs' / ('{}-{}-{}-{}.log'.format(environment, role, method or 'shared', seed))
        job['log'] = str(log)
        start = time.perf_counter()
        print('RUN {} {} {} seed{}'.format(environment, role, method or 'shared', seed), flush=True)
        try:
            explicit = args.mujoco131 if environment == 'walker-rand-params' else args.mujoco150
            env = worker_environment(environment, explicit)
            command = 'env ' + ' '.join(shlex.quote(k + '=' + env[k]) for k in ('MUJOCO_PY_MJPRO_PATH', 'LD_LIBRARY_PATH'))
            command += ' ' + shlex.join(cmd)
            with (out / 'commands.sh').open('a', encoding='utf-8') as stream:
                stream.write(command + '\n')
            job['command'] = command
            with log.open('w', encoding='utf-8') as stream:
                result = subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
            job['exit_code'] = result.returncode
            receipt = destination / ('manifest.json' if role == 'bank' else 'completion.json')
            if result.returncode != 0 or not receipt.is_file():
                raise RuntimeError('Worker failed; see ' + str(log))
            data = json.loads(receipt.read_text(encoding='utf-8'))
            if role != 'bank':
                expected = json.loads((bank / 'manifest.json').read_text(encoding='utf-8'))
                if not data['smoke_passed'] or data['raw_context_hashes'] != [c['context_hash'] for c in expected['contexts']]:
                    raise ValueError('Worker did not verify the identical shared raw contexts')
            job.update(passed=True, receipt_sha256=sha256(receipt))
        except Exception as exc:
            job['error'] = str(exc)
        job['elapsed_seconds'] = time.perf_counter() - start
        jobs.append(job)
        write_json(out / 'jobs.json', jobs)
        print('{} ({:.1f}s) {}'.format('PASS' if job['passed'] else 'FAIL', job['elapsed_seconds'], log), flush=True)
        return job['passed']

    for environment in PROTOCOL['environments']:
        bank = out / 'banks' / environment
        if not launch(environment, 'bank', bank):
            continue
        for entry in selection['models'] + selection['collectors']:
            if entry['environment'] != environment:
                continue
            role = entry.get('role', 'evaluation')
            destination = out / role / environment / entry['method'] / ('seed' + str(entry['training_seed']))
            launch(environment, role, destination, entry['method'], entry['training_seed'], bank)
    verify_inputs(plan)
    if any(sha256(path) != expected for path, expected in diagnostic_sources.items()):
        raise ValueError('Diagnostic implementation changed during smoke')
    # Check small run configurations as well; model hashes were checked on each load.
    for entry in selection['models'] + selection['collectors']:
        item = entry['config_file']
        if sha256(item['path']) != item['sha256']:
            raise ValueError('Run configuration changed during smoke: ' + item['path'])
    complete = stage_complete(selection, jobs, True)
    passed = sum(j['passed'] and j['role'] == 'evaluation' for j in jobs)
    selection['complete'] = complete
    for entry in selection['models'] + selection['collectors']:
        job = next((j for j in jobs if j['role'] == entry.get('role', 'evaluation') and j['environment'] == entry['environment']
                    and j['method'] == entry['method'] and j['seed'] == entry['training_seed']), None)
        entry['status'] = 'strict_load_and_real_smoke_passed' if job and job['passed'] else 'smoke_failed_or_not_run'
        entry['validation_output'] = job['output'] if job else None
    write_json(out / 'models.json', selection)
    result = dict(stage_A_complete=complete, formal_result=False, evaluation_models_passed=passed,
                  evaluation_models_required=64, collectors200_passed=sum(j['passed'] and j['role'] == 'collector200' for j in jobs),
                  unavailable=selection['unavailable'], failed_jobs=[j for j in jobs if not j['passed']],
                  input_verification=checked, next_step='Review this receipt, then implement/run B' if complete else 'Resolve unavailable/failed jobs; remain in A',
                  provenance_limit='Legacy GENTLE/Ours state_dicts have no embedded training seed/source signature; same-run pairs and native data reconstruction are recorded')
    write_json(out / 'completion.json', result)
    (out / 'STATUS.md').write_text('# Stage A: {}\n\n'.format('COMPLETE' if complete else 'IN_PROGRESS') +
        'Protocol: shift-diagnostics-v3. Smoke only; no formal B/C/D results.\n\n' +
        'Evaluation models passed: {}/64. Collector200 passed: {}/2.\n\n'.format(passed, result['collectors200_passed']) +
        'Commands: commands.sh; outputs and exit codes: jobs.json; validation: completion.json.\n\n' +
        'Next: ' + result['next_step'] + '\n', encoding='utf-8')
    root_status = Path(__file__).resolve().parent.parent / 'STATUS.md'
    with root_status.open('a', encoding='utf-8') as stream:
        stream.write('\n\n## Server stage-A execution: ' + datetime.now(timezone.utc).isoformat() + '\n\n' +
            'Status: {}. Exit code: {}. Evaluation models: {}/64; collector200: {}/2.\n\n'.format(
                'COMPLETE' if complete else 'IN_PROGRESS', 0 if complete else 2, passed, result['collectors200_passed']) +
            'Output: `' + str(out) + '`. Actual commands: `commands.sh`. Validation: `completion.json`, `jobs.json`.\n\n' +
            'Next: ' + result['next_step'] + '.\n')
    return 0 if complete else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path(__file__).with_name('stage_a_plan.json'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--mujoco131')
    parser.add_argument('--mujoco150')
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = (args.out or Path('/root/tievnas2/YinCH/shift-diagnostics-v3/smoke') / ('A-{}-{}'.format(stamp, os.getpid()))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'commands.sh').write_text('#!/usr/bin/env bash\n# Coordinator invocation:\n# ' + shlex.join([sys.executable] + sys.argv) + '\n', encoding='utf-8')
    (out / 'STATUS.md').write_text('Stage A: RUNNING. No completion receipt yet.\n', encoding='utf-8')
    print('Output directory:', out, flush=True)
    try:
        code = run(args, out)
    except Exception as exc:
        write_json(out / 'failure.json', dict(stage_A_complete=False, error=str(exc), traceback=traceback.format_exc()))
        (out / 'STATUS.md').write_text('Stage A: IN_PROGRESS; coordinator failed. See failure.json and commands.sh.\n', encoding='utf-8')
        traceback.print_exc()
        code = 1
    print('Stage A {}. Download the entire output directory: {}'.format('COMPLETE' if code == 0 else 'NOT complete', out), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
