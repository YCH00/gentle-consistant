"""Export one real-data shared-main-v1 bundle without starting training."""
import argparse
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env', required=True, choices=('point-robot', 'ant-dir', 'cheetah-vel',
                                                  'cheetah-dir', 'hopper-rand-params', 'walker-rand-params'))
    p.add_argument('--data-dir', required=True, type=Path, help='Raw directory for this environment')
    p.add_argument('--output-root', required=True, type=Path, help='Publishes <output-root>/<env> atomically')
    p.add_argument('--data-actions', required=True, choices=('normalized', 'physical'))
    p.add_argument('--action-evidence', required=True, help='How the collection action coordinates were verified')
    p.add_argument('--normalizer', type=Path, help='Optional validated existing NPZ; copied byte-for-byte')
    p.add_argument('--task-manifest', type=Path, help='Optional validated existing task JSON; copied byte-for-byte')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from shared_protocol.export import export_environment, write_report
    from shared_protocol.common import write_json
    try:
        target, status = export_environment(args.env, args.data_dir, args.output_root,
            args.data_actions, args.action_evidence, args.normalizer, args.task_manifest)
    except Exception as error:
        # Only write a failure report outside raw data, never label staging output complete.
        out, data = args.output_root.expanduser().resolve(), args.data_dir.expanduser().resolve()
        if out != data and data not in out.parents:
            write_json(out / (args.env + '.failure.json'), dict(environment=args.env, status='failed',
                                                              error='{}: {}'.format(type(error).__name__, error)))
            write_report(out)
        raise
    write_report(args.output_root.expanduser().resolve())
    print('{}: {}'.format(status, target))


if __name__ == '__main__':
    main()
