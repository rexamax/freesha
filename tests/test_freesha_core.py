import json
import tempfile
import unittest
from pathlib import Path

from freesha_core import (
    FreeshaOptimizer,
    LocalContextCache,
    SavingsLedger,
    TaskStore,
)


class FreeshaOptimizerTests(unittest.TestCase):
    def test_json_optimization_is_lossless_and_smaller(self):
        payload = {
            "task": "classify",
            "items": [{"id": i, "text": "same signal"} for i in range(20)],
            "metadata": {"source": "telegram", "valid": True},
        }
        source = json.dumps(payload, indent=4)
        result = FreeshaOptimizer().optimize_json(source)

        self.assertEqual(json.loads(result.optimized), payload)
        self.assertLess(len(result.optimized), len(source))
        self.assertGreater(result.bytes_saved, 0)
        self.assertEqual(result.mode, "json-minify")

    def test_payload_receipt_reports_input_and_output_tokens(self):
        payload = {
            "model": "gpt-5.6",
            "messages": [{"role": "user", "content": json.dumps({"a": 1}, indent=2)}],
        }
        result = FreeshaOptimizer().optimize_payload(payload)

        self.assertIn("messages", result.payload)
        self.assertGreater(result.receipt["input_tokens_before"], 0)
        self.assertGreaterEqual(result.receipt["input_tokens_after"], 0)
        self.assertIn("transformations", result.receipt)
        self.assertIn("estimated_input_reduction_pct", result.receipt)

    def test_python_skeleton_is_explicit_and_keeps_signatures(self):
        source = """
class Worker:
    def run(self, item: str) -> bool:
        return bool(item)

def parse(value: str, limit: int = 3) -> list[str]:
    return value.split()[:limit]
"""
        result = FreeshaOptimizer().python_skeleton(source)
        self.assertIn("class Worker", result)
        self.assertIn("def run(self, item: str) -> bool", result)
        self.assertIn("def parse(value: str, limit: int=3)", result)


class CacheAndTasksTests(unittest.TestCase):
    def test_context_cache_reuses_identical_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = LocalContextCache(Path(tmp) / "cache.json")
            first = cache.remember("same content")
            second = cache.remember("same content")
            self.assertFalse(first.hit)
            self.assertTrue(second.hit)
            self.assertEqual(first.key, second.key)

    def test_task_store_creates_and_updates_local_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TaskStore(Path(tmp) / "tasks.json")
            task = store.add("Prepare Build Week demo", priority="high")
            self.assertEqual(task.status, "todo")
            updated = store.update(task.id, status="done")
            self.assertEqual(updated.status, "done")
            self.assertEqual(store.list()[0].title, "Prepare Build Week demo")

    def test_savings_ledger_persists_receipts_and_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = SavingsLedger(Path(tmp) / "receipts.jsonl")
            ledger.record({"input_tokens_before": 100, "input_tokens_after": 60})
            ledger.record({"input_tokens_before": 20, "input_tokens_after": 10})
            self.assertEqual(ledger.summary(), {
                "requests": 2,
                "tokens_before": 120,
                "tokens_after": 70,
                "tokens_saved": 50,
                "reduction_pct": 41.67,
            })


if __name__ == "__main__":
    unittest.main()
