"""Tests of evidence export, not RL smoke or experimental results."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from audit import sha256, write_json
from export_evidence import approved_path, export
from review_audit import verify_audit


class EvidenceTests(unittest.TestCase):
    def test_export_omits_weights_and_reports_metadata_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            repo = base / 'method'
            run = repo / 'runs/cheetah-vel/seed0'
            run.mkdir(parents=True)
            (repo / 'loader.py').write_text('# source', encoding='utf-8')
            (run / 'last.pt').write_bytes(b'never load or copy this checkpoint')
            config = run / 'resolved_config.json'
            config.write_text('{"iteration": 499}', encoding='utf-8')
            audit = base / 'audit'
            audit.mkdir()
            candidate = {'path': str(config), 'kind': 'shared_or_run_metadata', 'sha256': sha256(config)}
            write_json(audit / 'candidates.json', [candidate])
            for name in ('resource_audit.json', 'source_snapshots.json', 'models.json', 'protocol.json'):
                write_json(audit / name, {})
            write_json(audit / 'audit_completion.json', {'output_sha256': {p.name: sha256(p) for p in audit.iterdir()}})
            request = {'server_base': str(base), 'audit_candidates_sha256': sha256(audit / 'candidates.json'),
                       'source_roots': [str(repo)], 'metadata_files': [candidate], 'checkpoint_inventory': []}
            config.write_text('{"iteration": 500}', encoding='utf-8')
            out = base / 'export'
            result = export(request, audit, out)
            self.assertEqual(result['errors'], [])
            self.assertEqual(result['changed_metadata'], [str(config)])
            with zipfile.ZipFile(out / 'evidence.zip') as archive:
                self.assertIn('files/method/loader.py', archive.namelist())
                self.assertIn('files/method/runs/cheetah-vel/seed0/resolved_config.json', archive.namelist())
                self.assertFalse(any(n.endswith('.pt') for n in archive.namelist()))
                for record in result['files']:
                    import hashlib
                    self.assertEqual(hashlib.sha256(archive.read(record['member'])).hexdigest(), record['sha256'])
            with self.assertRaises(ValueError):
                export(request, audit, out)
            (audit / 'models.json').write_text('{}\nchanged', encoding='utf-8')
            with self.assertRaises(ValueError):
                verify_audit(audit)

    def test_paths_cannot_escape_server_base(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve() / 'allowed'
            base.mkdir()
            with self.assertRaises(ValueError):
                approved_path(base / '../outside.py', base)


if __name__ == '__main__':
    unittest.main()
