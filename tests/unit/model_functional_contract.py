"""CPU-only negative contracts for the complete-checkpoint evidence verifier."""
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration/tooling"))
from model_functional import CASES, verify


class ModelFunctionalContract(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.evidence = Path(self.temporary.name)
        for variant in ("eager", "graph", "chunk", "graph-chunk"):
            chunk = "chunk" in variant
            records, targets = [], []
            for case in CASES:
                identities = [str(i) for i in reversed(range(len(case.requests)))]
                first = [{"id": rid, "tokens": 1, "computed_before": 0,
                          "prompt": case.requests[int(rid)].prompt} for rid in identities]
                second = [{**item, "computed_before": 1 if chunk else item["prompt"]} for item in first]
                samples = {rid: [100 + int(rid)] for rid in identities
                           if not (chunk and case.name == "cancel-prefill" and rid == "0")}
                values = [[100 + int(rid)] if rid in samples else [] for rid in identities]
                records.append({"case": asdict(case), "steps": [{"items": first, "samples": {}},
                                {"items": second, "samples": samples}], "token_ids": samples,
                                "graph_replays": int("graph" in variant)})
                targets.extend(((first, [[] for _ in identities]), (second, values)))
            examples = [{"prompt_token_ids": [1, 2], "token_ids": [3], "text": "fixture"}]
            document = {"tp": 2, "ep": False, "source_sha256": "fixture-collector",
                        "identity": {"model_config_sha256": "fixture-config"}, "parameter_bytes": 4096,
                        "records": records, "examples": examples}
            summary = {"cases": 6, "steps": 12, "control_reads": 42, "cuda_initialized": False,
                       "target_sha256": hashlib.sha256(json.dumps(targets, sort_keys=True).encode()).hexdigest()}
            directory = self.evidence / variant
            directory.mkdir()
            for rank in (0, 1):
                self.write(directory / f"rank-{rank}-native.json", document)
                self.write(directory / f"rank-{rank}-replay.json", summary)
        self.write(self.evidence / "cpu-reference.json", {"examples": examples, "cuda_initialized": False})

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, path, document):
        path.write_text(json.dumps(document, sort_keys=True))

    def check(self, rank=0):
        with redirect_stdout(io.StringIO()):
            verify(SimpleNamespace(evidence=self.evidence, rank=rank, tp=2, ep=False))

    def comparison(self):
        return json.loads((self.evidence / "comparison.json").read_text())

    def test_valid_summaries_on_both_ranks_without_framework_imports(self):
        for rank in (0, 1):
            self.check(rank)
            self.assertTrue(self.comparison()["original_scheduler_replay_verified"])
            self.assertTrue(self.comparison()["native_recorded_outputs_verified"])
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("vllm", sys.modules)

    def test_incomplete_or_stale_summary_fields_are_rejected_and_saved(self):
        path = self.evidence / "graph/rank-0-replay.json"
        original = json.loads(path.read_text())
        for field, value in (("cases", 0), ("steps", 0), ("control_reads", 0),
                             ("target_sha256", "0" * 64), ("cuda_initialized", True)):
            with self.subTest(field=field):
                self.write(path, {**original, field: value})
                with self.assertRaises(AssertionError):
                    self.check()
                comparison = self.comparison()
                self.assertFalse(comparison["original_scheduler_replay_verified"])
                self.assertEqual(comparison["replay_integrity_differences"],
                                 [{"variant": "graph", "field": field,
                                   "expected": original[field], "observed": value}])
        self.write(path, original)

    def test_summary_from_another_policy_is_rejected(self):
        source = json.loads((self.evidence / "chunk/rank-0-replay.json").read_text())
        self.write(self.evidence / "graph/rank-0-replay.json", source)
        with self.assertRaises(AssertionError):
            self.check()
        self.assertEqual([d["field"] for d in self.comparison()["replay_integrity_differences"]],
                         ["target_sha256"])

    def test_changed_native_samples_invalidate_equal_count_summary(self):
        path = self.evidence / "graph/rank-1-native.json"
        document = json.loads(path.read_text())
        document["records"][0]["steps"][1]["samples"]["0"] = [999]
        self.write(path, document)
        with self.assertRaises(AssertionError):
            self.check(rank=1)
        self.assertEqual([d["field"] for d in self.comparison()["replay_integrity_differences"]],
                         ["target_sha256"])

    def test_removed_cancelled_output_retains_strict_numerical_failure(self):
        path = self.evidence / "graph/rank-0-native.json"
        document = json.loads(path.read_text())
        record = next(r for r in document["records"] if r["case"]["name"] == "cancel-decode")
        del record["token_ids"]["0"]
        self.write(path, document)
        with self.assertRaises(AssertionError):
            self.check()
        comparison = self.comparison()
        self.assertTrue(comparison["original_scheduler_replay_verified"])
        self.assertFalse(comparison["native_recorded_outputs_verified"])
        self.assertFalse(comparison["graph_matched_policy_exact"])
        self.assertEqual(comparison["matched_policy_differences"][0]["request"], "0")

    def test_coherent_output_edits_cannot_hide_sampled_tokens(self):
        for pair in (("eager", "graph"), ("chunk", "graph-chunk")):
            for rank in (0, 1):
                for tokens in (None, [999]):
                    with self.subTest(pair=pair, rank=rank, tokens=tokens):
                        originals = {}
                        for variant in pair:
                            path = self.evidence / variant / f"rank-{rank}-native.json"
                            originals[path] = path.read_text()
                            document = json.loads(originals[path])
                            record = next(r for r in document["records"] if r["case"]["name"] == "cancel-decode")
                            if tokens is None:
                                del record["token_ids"]["0"]
                            else:
                                record["token_ids"]["0"] = tokens
                            self.write(path, document)
                        with self.assertRaises(AssertionError):
                            self.check(rank)
                        comparison = self.comparison()
                        self.assertTrue(comparison["graph_matched_policy_exact"])
                        self.assertTrue(comparison["original_scheduler_replay_verified"])
                        self.assertFalse(comparison["native_recorded_outputs_verified"])
                        self.assertEqual(comparison["native_recorded_output_differences"],
                                         [{"variant": variant, "case": "cancel-decode", "request": "0",
                                           "sampled_tokens": [100], "reported_tokens": tokens or []} for variant in pair])
                        for path, content in originals.items():
                            path.write_text(content)


if __name__ == "__main__":
    unittest.main()
