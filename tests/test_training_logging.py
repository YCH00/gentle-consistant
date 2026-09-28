"""Exercise logging failure paths without importing Gym, MuJoCo or pandas."""

import ast
import contextlib
import csv
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_function(relative_path, name, namespace):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class TrainingLoggingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'progress.csv'
        self.handle = self.path.open('w', newline='', encoding='utf-8')
        self.addCleanup(self.handle.close)
        self.state = dict(
            csv=csv, _tabular=[], _log_tabular_only=False,
            _tabular_fds={'progress': self.handle}, _tabular_header_written=set(),
            tabulate=lambda rows: '', log=lambda *args, **kwargs: None,
        )
        self.dump = load_function('rlkit/core/logger.py', 'dump_tabular', self.state)

    def rows(self):
        with self.path.open(newline='', encoding='utf-8') as source:
            return list(csv.DictReader(source))

    def record(self, epoch):
        self.state['_tabular'].extend([
            ('AverageReturn_all_test_tasks_expl', 123.5),
            ('QF Loss', 2.0), ('Policy Mean', 0.25), ('coverage', 0.8),
            ('Epoch', epoch),
        ])

    def test_every_epoch_is_flushed_with_unchanged_tags_and_steps(self):
        events, flushes = [], []

        def add_scalar(tag, value, epoch):
            # Reading a separate handle also verifies the CSV was flushed first.
            self.assertEqual(self.rows()[-1]['Epoch'], str(epoch))
            events.append((tag, value, epoch))

        writer = SimpleNamespace(add_scalar=add_scalar,
                                 flush=lambda: flushes.append(len(events)))
        for epoch in (7, 8):
            self.record(epoch)
            self.dump(tb_writer=writer)
        self.assertEqual([r['Epoch'] for r in self.rows()], ['7', '8'])
        self.assertEqual(events[:4], [
            ('Return/AverageReturn_all_test_tasks_expl', 123.5, 7),
            ('Loss/QF Loss', 2.0, 7), ('Policy/Policy Mean', 0.25, 7),
            ('Other/coverage', 0.8, 7),
        ])
        self.assertEqual([e[2] for e in events[4:]], [8] * 4)
        self.assertEqual(flushes, [4, 8])
        self.assertEqual(self.state['_tabular'], [])

    def check_writer_failure(self, stage):
        error = OSError('event output unavailable')

        def fail(*args):
            raise error

        writer = SimpleNamespace(add_scalar=lambda *args: None, flush=lambda: None)
        setattr(writer, stage, fail)
        self.record(49)
        with self.assertRaises(OSError) as caught:
            self.dump(tb_writer=writer)
        self.assertIs(caught.exception, error)
        self.assertEqual(self.rows()[0]['Epoch'], '49')
        self.assertEqual(self.state['_tabular'], [])

    def test_csv_survives_tensorboard_add_failure(self):
        self.check_writer_failure('add_scalar')

    def test_csv_survives_tensorboard_flush_failure(self):
        self.check_writer_failure('flush')

    def test_csv_only_logging(self):
        self.record(49)
        self.dump()
        self.assertEqual(self.rows()[0]['Epoch'], '49')

    def run_training(self, error=None):
        events, output = [], io.StringIO()
        writer = SimpleNamespace(close=lambda: events.append('closed'))

        def train(received):
            self.assertIs(received, writer)
            events.append('trained')
            if error is not None:
                raise error

        train_with_writer = load_function(
            'train_gentle.py', '_train_with_tensorboard',
            {'SummaryWriter': lambda log_dir: writer},
        )
        with contextlib.redirect_stdout(output):
            if error is None:
                train_with_writer(SimpleNamespace(train=train), 'run')
            else:
                with self.assertRaises(type(error)) as caught:
                    train_with_writer(SimpleNamespace(train=train), 'run')
                self.assertIs(caught.exception, error)
        self.assertEqual(events, ['trained', 'closed'])
        self.assertEqual('Training completed:' in output.getvalue(), error is None)

    def test_writer_closes_after_success(self):
        self.run_training()

    def test_writer_closes_after_exception_or_interrupt(self):
        for error in (RuntimeError('training failed'), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                self.run_training(error)


if __name__ == '__main__':
    unittest.main()
