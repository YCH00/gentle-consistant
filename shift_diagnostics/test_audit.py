"""Resource-audit guardrail tests. Fixtures are not experimental data."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import audit


class AuditTests(unittest.TestCase):
    def test_all_required_model_slots_are_preserved(self):
        models = audit.blank_models()["models"]
        self.assertEqual(len(models), 66)
        self.assertEqual(sum("evaluation" in m["roles"] for m in models), 64)
        self.assertEqual(sum("collector" in m["roles"] for m in models), 4)
        self.assertEqual({m["method"] for m in models}, set(audit.METHODS))
        self.assertTrue(all(m["status"] in ("missing", "pending_training") for m in models))
        self.assertEqual(sum(m["status"] == "pending_training" for m in models), 4)
        self.assertEqual({m['user_selected_run_directory'] for m in models if m['method'] == 'Ours-local'}, {'23_local_decay'})

    def test_protocol_is_bound_to_handoff(self):
        root = Path(__file__).resolve().parent.parent
        protocol = json.loads((root / "shift_diagnostics/protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(audit.sha256(root / protocol["specification"]), protocol["specification_sha256"])
        self.assertEqual(protocol["methods"], audit.METHODS)
        self.assertFalse(protocol["component_ablations"])
        self.assertEqual((protocol["context_length"], protocol["contexts_per_task_condition"],
                          protocol["rollouts_per_context"]), (200, 20, 3))

    def test_candidates_are_not_auto_accepted_or_deserialized(self):
        with tempfile.TemporaryDirectory(prefix="shift-audit-test-") as temp:
            base = Path(temp)
            repo = base / "repo"
            repo.mkdir()
            (repo / "policy_itr_499.pth").write_bytes(b"not a pickle; must never load")
            (repo / "policy_itr_200.pth").write_bytes(b"collector candidate")
            (repo / "policy_itr_100.pth").write_bytes(b"wrong iteration")
            tests = repo / "output/shared-artifact-tests/old"
            tests.mkdir(parents=True)
            (tests / "policy_itr_499.pth").write_bytes(b"excluded test")
            envs = repo / "rlkit/envs"
            envs.mkdir(parents=True)
            (envs / "real_environment.py").write_text("# source evidence", encoding="utf-8")
            out = base / "audit"
            args = ["--repo-root", str(repo), "--search-root", str(base),
                    "--search-root", str(base / "missing-shared"), "--out", str(out), "--hash-checkpoints"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(args), 0)
            candidates = json.loads((out / "candidates.json").read_text(encoding="utf-8"))
            self.assertEqual(len(candidates), 2)
            self.assertTrue(all(c["sha256"] and c["assignment"] == "unverified" for c in candidates))
            report = json.loads((out / "resource_audit.json").read_text(encoding="utf-8"))
            self.assertFalse(report["stage_complete"])
            self.assertTrue(any(r["status"] == "missing" for r in report["requested_roots"]))
            snapshot = json.loads((out / "source_snapshots.json").read_text(encoding="utf-8"))
            self.assertIn("rlkit/envs/real_environment.py", [f["path"] for f in snapshot[0]["files"]])
            completion = json.loads((out / "audit_completion.json").read_text(encoding="utf-8"))
            self.assertFalse(completion["stage_A_complete"])
            for name, expected in completion["output_sha256"].items():
                self.assertEqual(audit.sha256(out / name), expected)
            before = audit.sha256(out / "models.json")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                audit.main(args)
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(audit.sha256(out / "models.json"), before)


if __name__ == "__main__":
    unittest.main()
