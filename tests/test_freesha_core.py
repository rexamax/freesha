import http.client
import io
import json
import os
import shutil
import threading
import unittest
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from freesha_core import (
    ContentStore,
    EconomyContextPlanner,
    FreeshaOptimizer,
    LocalContextCache,
    SavingsLedger,
    TaskStore,
    ToolOutputCompactor,
    _exclusive_file_lock,
    _NoRedirectHandler,
    forward_chat,
    run_benchmark,
)


@contextmanager
def writable_temp_directory():
    """Create test state without Windows tempfile ACLs blocking child processes."""
    root = (Path.cwd() / ".test-state" / "unit").resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = (root / uuid.uuid4().hex).resolve()
    if path.parent != root:
        raise RuntimeError("refusing to create test state outside .test-state/unit")
    path.mkdir()
    try:
        yield str(path)
    finally:
        if path.parent != root:
            raise RuntimeError("refusing to remove test state outside .test-state/unit")
        shutil.rmtree(path)


class FreeshaOptimizerTests(unittest.TestCase):
    def test_json_optimization_is_lossless_and_smaller(self):
        payload = {
            "task": "classify",
            "items": [{"id": i, "text": "same signal"} for i in range(20)],
            "metadata": {"source": "event-stream", "valid": True},
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

    def test_openai_cache_preparation_preserves_request_and_adds_supported_fields(self):
        payload = {
            "model": "gpt-5.6",
            "messages": [
                {"role": "system", "content": "Stable instructions"},
                {"role": "user", "content": "Variable question"},
            ],
            "temperature": 0.2,
        }
        result = FreeshaOptimizer().prepare_openai_cache(payload, cache_key="support:v1")

        self.assertEqual(result.payload["messages"], payload["messages"])
        self.assertEqual(result.payload["temperature"], 0.2)
        self.assertEqual(result.payload["prompt_cache_key"], "support:v1")
        self.assertEqual(
            result.payload["prompt_cache_options"],
            {"mode": "implicit", "ttl": "30m"},
        )
        self.assertNotIn("metadata", result.payload)
        self.assertIn("openai-prompt-cache", result.receipt["transformations"])

    def test_optimize_request_does_not_inject_unknown_metadata(self):
        payload = {
            "model": "gpt-5.6",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = FreeshaOptimizer().optimize_request(payload)
        self.assertNotIn("metadata", result.payload)

    def test_forward_chat_rejects_insecure_remote_endpoint_before_sending_key(self):
        endpoints = [
            "http://example.com/v1/chat/completions",
            "file:///tmp/fake-endpoint",
            "ftp://example.com/chat",
            "http://localhost.example.com/chat",
        ]
        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint), self.assertRaisesRegex(ValueError, "HTTPS"):
                forward_chat(
                    {"model": "gpt-5.6", "messages": []},
                    "synthetic-test-key",
                    endpoint=endpoint,
                )

    def test_forward_chat_allows_loopback_http_after_validation(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"ok": true}'

        with patch("freesha_core._open_without_redirects", return_value=Response()) as send:
            result = forward_chat(
                {"model": "gpt-5.6", "messages": []},
                "synthetic-test-key",
                endpoint="http://127.0.0.1:8080/v1/chat/completions",
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(send.call_count, 1)

    def test_forward_chat_rejects_every_redirect_hop(self):
        cases = [
            ("https://safe.example/start", "http://attacker.example/steal"),
            ("https://safe.example/start", "https://other.example/steal"),
            ("http://127.0.0.1/start", "http://attacker.example/steal"),
            ("https://safe.example/start", "https://safe.example/next"),
        ]
        handler = _NoRedirectHandler()
        for source, target in cases:
            with self.subTest(source=source, target=target):
                request = urllib.request.Request(source)
                redirected = handler.redirect_request(
                    request,
                    io.BytesIO(),
                    302,
                    "Found",
                    http.client.HTTPMessage(),
                    target,
                )
                self.assertIsNone(redirected)

    def test_forward_chat_does_not_follow_redirects_with_authorization(self):
        received_authorization: list[str | None] = []

        class TargetHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                received_authorization.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"unexpected": true}')

            def log_message(self, format, *args):
                pass

        target = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)
        target_thread = threading.Thread(target=target.serve_forever, daemon=True)
        target_thread.start()
        target_url = f"http://127.0.0.1:{target.server_port}/steal"

        class RedirectHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(302)
                self.send_header("Location", target_url)
                self.end_headers()

            def log_message(self, format, *args):
                pass

        redirect = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
        redirect_thread = threading.Thread(target=redirect.serve_forever, daemon=True)
        redirect_thread.start()

        try:
            with self.assertRaisesRegex(RuntimeError, "upstream HTTP 302"):
                forward_chat(
                    {"model": "gpt-5.6", "messages": []},
                    "synthetic-test-key",
                    endpoint=f"http://127.0.0.1:{redirect.server_port}/start",
                )
        finally:
            redirect.shutdown()
            redirect.server_close()
            target.shutdown()
            target.server_close()
            redirect_thread.join(timeout=2)
            target_thread.join(timeout=2)

        self.assertEqual(received_authorization, [])


class RecoverableEconomyModeTests(unittest.TestCase):
    def test_repetitive_tool_output_is_compacted_and_exactly_recoverable(self):
        lines = [
            f"2026-07-18T12:00:{i:02d}Z INFO worker job={1000 + i} completed in {i + 1}ms"
            for i in range(80)
        ]
        lines.insert(
            41,
            "2026-07-18T12:00:41Z ERROR request_id=req-critical database timeout",
        )
        source = "\n".join(lines)

        with writable_temp_directory() as tmp:
            store = ContentStore(Path(tmp) / "blobs")
            result = ToolOutputCompactor(store).compact(source)

            self.assertTrue(result.changed)
            self.assertGreaterEqual(result.reduction_pct, 70.0)
            self.assertIn("ERROR request_id=req-critical database timeout", result.output)
            self.assertIn("[repeated x", result.output)
            self.assertIsNotNone(result.recovery_key)
            self.assertEqual(store.get(result.recovery_key), source)

    def test_short_output_uses_net_loss_passthrough(self):
        with writable_temp_directory() as tmp:
            result = ToolOutputCompactor(ContentStore(Path(tmp) / "blobs")).compact("ok")
            self.assertFalse(result.changed)
            self.assertEqual(result.output, "ok")
            self.assertIsNone(result.recovery_key)

    def test_net_loss_does_not_leave_orphan_recovery_blob(self):
        with writable_temp_directory() as tmp:
            blob_path = Path(tmp) / "blobs"
            result = ToolOutputCompactor(ContentStore(blob_path), minimum_tokens=1).compact(
                "a\na\na"
            )

            self.assertFalse(result.changed)
            self.assertEqual(result.mode, "net-loss-passthrough")
            self.assertFalse(blob_path.exists())

    def test_content_store_concurrent_put_is_safe(self):
        content = "recoverable event payload"
        with writable_temp_directory() as tmp:
            store = ContentStore(Path(tmp) / "blobs")
            with ThreadPoolExecutor(max_workers=16) as executor:
                keys = list(executor.map(store.put, [content] * 64))
            self.assertEqual(set(keys), {keys[0]})
            self.assertEqual(store.get(keys[0]), content)

    def test_content_store_preserves_crlf_exactly(self):
        content = "first\r\nsecond\rthird\n"
        with writable_temp_directory() as tmp:
            store = ContentStore(Path(tmp) / "blobs")
            self.assertEqual(store.get(store.put(content)), content)

    def test_content_store_rejects_tampered_existing_blob(self):
        with writable_temp_directory() as tmp:
            blob_path = Path(tmp) / "blobs"
            store = ContentStore(blob_path)
            key = store.put("original")
            (blob_path / f"{key}.txt").write_text("tampered", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "integrity"):
                store.get(key)
            with self.assertRaisesRegex(ValueError, "integrity"):
                store.put("original")

    def test_compactor_preserves_run_order_around_critical_event(self):
        routine = "2026-07-18T12:00:00Z INFO worker job=1000 completed in 10ms"
        later = "2026-07-18T12:00:05Z INFO worker job=1005 completed in 15ms"
        critical = "2026-07-18T12:00:04Z ERROR request_id=req-7 timeout"
        source = "\n".join(([routine] * 24) + [critical] + ([later] * 18))

        with writable_temp_directory() as tmp:
            store = ContentStore(Path(tmp) / "blobs")
            result = ToolOutputCompactor(store).compact(source)
            recovered = store.get(result.recovery_key)

        self.assertTrue(result.changed)
        self.assertNotEqual(result.output, source)
        self.assertLess(result.output.index(routine), result.output.index(critical))
        self.assertLess(result.output.index(critical), result.output.index(later))
        self.assertIn("[repeated x24", result.output)
        self.assertIn("[repeated x18", result.output)
        self.assertIsNotNone(result.recovery_key)
        self.assertEqual(recovered, source)

    def test_context_planner_deduplicates_and_keeps_required_and_relevant_items(self):
        packet = {
            "task": "Fix payment timeout request req-42",
            "items": [
                {
                    "id": "contract",
                    "content": "Never change the public API.",
                    "required": True,
                },
                {"id": "noise-a", "content": "Unrelated design color discussion."},
                {
                    "id": "duplicate",
                    "content": "Unrelated design color discussion.",
                },
                {
                    "id": "incident",
                    "content": "ERROR req-42 payment timeout in checkout worker",
                },
            ],
        }
        with writable_temp_directory() as tmp:
            result = EconomyContextPlanner(content_store=ContentStore(Path(tmp) / "blobs")).build(
                packet, token_budget=40
            )

            self.assertIn("[contract]", result.context)
            self.assertIn("[incident]", result.context)
            self.assertEqual(result.receipt["duplicates_removed"], 1)
            self.assertGreater(
                result.receipt["input_tokens_before"],
                result.receipt["input_tokens_after"],
            )
            self.assertIn("contract", result.receipt["selected_ids"])
            self.assertIn("incident", result.receipt["selected_ids"])

    def test_context_planner_rejects_private_paths_in_public_ids(self):
        packet = {
            "task": "Summarize",
            "items": [{"id": "/root/private/note.md", "content": "safe generic text"}],
        }
        with writable_temp_directory() as tmp:
            planner = EconomyContextPlanner(ContentStore(Path(tmp) / "blobs"))
            with self.assertRaises(ValueError):
                planner.build(packet, token_budget=100)

    def test_context_planner_rejects_duplicate_item_ids(self):
        packet = {
            "task": "Compare events",
            "items": [
                {"id": "event", "content": "first event"},
                {"id": "event", "content": "second event"},
            ],
        }
        with writable_temp_directory() as tmp:
            planner = EconomyContextPlanner(ContentStore(Path(tmp) / "blobs"))
            with self.assertRaisesRegex(ValueError, "unique"):
                planner.build(packet, token_budget=100)

    def test_context_planner_rejects_non_boolean_required_flag(self):
        packet = {
            "task": "Respect budget",
            "items": [{"id": "oversized", "content": "noise " * 100, "required": "false"}],
        }
        with writable_temp_directory() as tmp:
            planner = EconomyContextPlanner(ContentStore(Path(tmp) / "blobs"))
            with self.assertRaisesRegex(ValueError, "required must be a boolean"):
                planner.build(packet, token_budget=10)

    def test_minified_json_context_is_byte_recoverable(self):
        original = '{\n  "request": "req-42",\n  "flags": [true, false]\n}'
        packet = {
            "task": "Inspect request",
            "items": [{"id": "payload", "kind": "json", "content": original}],
        }
        with writable_temp_directory() as tmp:
            store = ContentStore(Path(tmp) / "blobs")
            result = EconomyContextPlanner(store).build(packet, token_budget=100)
            recovery_key = result.receipt["recovery_keys"]["payload"]
            self.assertEqual(store.get(recovery_key), original)

        self.assertIn('{"request":"req-42","flags":[true,false]}', result.context)

    def test_json_minification_uses_net_loss_passthrough(self):
        original = '{"x":1e100}'
        packet = {
            "task": "Inspect number",
            "items": [{"id": "number", "kind": "json", "content": original}],
        }
        with writable_temp_directory() as tmp:
            result = EconomyContextPlanner(ContentStore(Path(tmp) / "blobs")).build(
                packet, token_budget=100
            )

        self.assertIn(f"[number]\n{original}", result.context)
        self.assertNotIn("number", result.receipt["recovery_keys"])

    def test_required_context_reserves_budget_before_optional_selection(self):
        task = " ".join(f"term{index}" for index in range(1001))
        required = {"id": "required", "content": "must preserve", "required": True}
        optional = {"id": "optional", "content": task}
        with writable_temp_directory() as tmp:
            planner = EconomyContextPlanner(ContentStore(Path(tmp) / "blobs"))
            optional_only = planner.build({"task": task, "items": [optional]}, token_budget=10_000)
            result = planner.build(
                {"task": task, "items": [required, optional]},
                token_budget=optional_only.receipt["input_tokens_after"],
            )

        self.assertIn("required", result.receipt["selected_ids"])
        self.assertNotIn("optional", result.receipt["selected_ids"])
        self.assertFalse(result.receipt["budget_exceeded"])

    def test_required_duplicate_wins_over_earlier_optional_copy(self):
        packet = {
            "task": "Preserve contract",
            "items": [
                {"id": "optional-copy", "content": "Keep the API stable."},
                {
                    "id": "required-contract",
                    "content": "Keep the API stable.",
                    "required": True,
                },
            ],
        }
        with writable_temp_directory() as tmp:
            result = EconomyContextPlanner(ContentStore(Path(tmp) / "blobs")).build(
                packet, token_budget=1
            )

        self.assertIn("required-contract", result.receipt["selected_ids"])
        self.assertNotIn("optional-copy", result.receipt["selected_ids"])
        self.assertTrue(result.receipt["quality_gates"]["required_items_preserved"])

    def test_case_distinct_context_is_not_deduplicated(self):
        packet = {
            "task": "Compare region labels",
            "items": [
                {"id": "upper", "content": "US"},
                {"id": "lower", "content": "us"},
            ],
        }
        with writable_temp_directory() as tmp:
            result = EconomyContextPlanner(ContentStore(Path(tmp) / "blobs")).build(
                packet, token_budget=100
            )

        self.assertEqual(result.receipt["duplicates_removed"], 0)
        self.assertEqual(set(result.receipt["selected_ids"]), {"upper", "lower"})

    def test_whitespace_sensitive_context_is_not_deduplicated(self):
        packet = {
            "task": "Compare Python branches",
            "items": [
                {"id": "one-space", "content": "if ready:\n result = 1"},
                {"id": "two-spaces", "content": "if ready:\n  result = 1"},
            ],
        }
        with writable_temp_directory() as tmp:
            result = EconomyContextPlanner(ContentStore(Path(tmp) / "blobs")).build(
                packet, token_budget=100
            )

        self.assertEqual(result.receipt["duplicates_removed"], 0)
        self.assertEqual(set(result.receipt["selected_ids"]), {"one-space", "two-spaces"})

    def test_all_required_items_survive_identical_content(self):
        packet = {
            "task": "Preserve contracts",
            "items": [
                {"id": "contract-a", "content": "Keep API stable.", "required": True},
                {"id": "contract-b", "content": "Keep API stable.", "required": True},
            ],
        }
        with writable_temp_directory() as tmp:
            result = EconomyContextPlanner(ContentStore(Path(tmp) / "blobs")).build(
                packet, token_budget=1
            )

        self.assertEqual(set(result.receipt["selected_ids"]), {"contract-a", "contract-b"})
        self.assertTrue(result.receipt["quality_gates"]["required_items_preserved"])

    def test_oversized_optional_item_does_not_bypass_budget(self):
        packet = {
            "task": "Find timeout",
            "items": [{"id": "huge", "content": "timeout " * 500}],
        }
        with writable_temp_directory() as tmp:
            store = ContentStore(Path(tmp) / "blobs")
            result = EconomyContextPlanner(store).build(packet, token_budget=20)
            recovery_key = result.receipt["recovery_keys"]["huge"]
            self.assertEqual(store.get(recovery_key), "timeout " * 500)

        self.assertNotIn("huge", result.receipt["selected_ids"])
        self.assertIn("huge", result.receipt["omitted_ids"])
        self.assertTrue(result.receipt["quality_gates"]["budget_respected_or_mandatory_overflow"])

    def test_canonical_benchmark_proves_savings_and_quality_gates(self):
        with writable_temp_directory() as tmp:
            report = run_benchmark(Path(tmp) / "blobs")

        self.assertGreaterEqual(report["aggregate"]["reduction_pct"], 70.0)
        self.assertTrue(report["quality_gates"]["all_passed"])
        self.assertTrue(report["quality_gates"]["json_equivalent"])
        self.assertTrue(report["quality_gates"]["log_exactly_recoverable"])
        self.assertTrue(report["quality_gates"]["critical_error_preserved"])
        self.assertEqual(report["network_calls"], 0)


class CacheAndTasksTests(unittest.TestCase):
    def test_context_cache_reuses_identical_content(self):
        with writable_temp_directory() as tmp:
            cache = LocalContextCache(Path(tmp) / "cache.json")
            first = cache.remember("same content")
            second = cache.remember("same content")
            self.assertFalse(first.hit)
            self.assertTrue(second.hit)
            self.assertEqual(first.key, second.key)

    def test_atomic_state_write_syncs_parent_directory(self):
        with writable_temp_directory() as tmp:
            path = Path(tmp) / "cache.json"
            with patch("freesha_core._fsync_directory") as sync_directory:
                LocalContextCache(path).remember("durable content")

            sync_directory.assert_called_once_with(path.parent)

    def test_lock_acquisition_failure_is_not_masked_by_unlock(self):
        lock_target = "msvcrt.locking" if os.name == "nt" else "fcntl.flock"
        with writable_temp_directory() as tmp:
            path = Path(tmp) / "cache.json"
            with (
                patch(lock_target, side_effect=OSError("acquire failed")) as lock_call,
                self.assertRaisesRegex(OSError, "acquire failed"),
                _exclusive_file_lock(path),
            ):
                self.fail("body must not run without the lock")

            self.assertEqual(lock_call.call_count, 1)

    def test_context_cache_preserves_malformed_state_and_fails_closed(self):
        with writable_temp_directory() as tmp:
            path = Path(tmp) / "cache.json"
            damaged = b'{"unfinished":'
            path.write_bytes(damaged)

            with self.assertRaisesRegex(ValueError, "malformed"):
                LocalContextCache(path).remember("new content")

            self.assertEqual(path.read_bytes(), damaged)

    def test_context_cache_rejects_invalid_nested_schema_without_overwrite(self):
        valid_key = "a" * 64
        cases = [
            {"x": {"tokens": 1, "last_seen": 1.0}},
            {valid_key: "not-a-cache-entry"},
            {valid_key: {"tokens": 1}},
            {valid_key: {"tokens": 1, "last_seen": 1.0, "extra": True}},
            {valid_key: {"tokens": True, "last_seen": 1.0}},
            {valid_key: {"tokens": -1, "last_seen": 1.0}},
            {valid_key: {"tokens": 1, "last_seen": False}},
            {valid_key: {"tokens": 1, "last_seen": float("nan")}},
        ]
        for index, value in enumerate(cases):
            with self.subTest(case=index), writable_temp_directory() as tmp:
                path = Path(tmp) / "cache.json"
                damaged = json.dumps(value).encode()
                path.write_bytes(damaged)

                with self.assertRaisesRegex(ValueError, "schema"):
                    LocalContextCache(path).remember("new content")

                self.assertEqual(path.read_bytes(), damaged)

    def test_context_cache_concurrent_instances_preserve_all_updates(self):
        with writable_temp_directory() as tmp:
            path = Path(tmp) / "cache.json"
            contents = [f"unique cache content {index}" for index in range(32)]

            def remember(content):
                return LocalContextCache(path).remember(content).key

            with ThreadPoolExecutor(max_workers=16) as executor:
                keys = list(executor.map(remember, contents))

            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(stored), set(keys))

    def test_task_store_creates_and_updates_local_task(self):
        with writable_temp_directory() as tmp:
            store = TaskStore(Path(tmp) / "tasks.json")
            task = store.add("Prepare Build Week demo", priority="high")
            self.assertEqual(task.status, "todo")
            updated = store.update(task.id, status="done")
            self.assertEqual(updated.status, "done")
            self.assertEqual(store.list()[0].title, "Prepare Build Week demo")

    def test_task_store_preserves_malformed_state_and_fails_closed(self):
        with writable_temp_directory() as tmp:
            path = Path(tmp) / "tasks.json"
            damaged = b"[not-json"
            path.write_bytes(damaged)

            with self.assertRaisesRegex(ValueError, "malformed"):
                TaskStore(path).add("new task")

            self.assertEqual(path.read_bytes(), damaged)

    def test_task_store_rejects_invalid_nested_schema_without_overwrite(self):
        valid = {
            "id": "a" * 10,
            "title": "task",
            "status": "todo",
            "priority": "normal",
            "created_at": 1.0,
            "updated_at": 2.0,
        }
        cases = [
            "not-a-task-row",
            {key: value for key, value in valid.items() if key != "status"},
            {**valid, "extra": True},
            {**valid, "id": 123},
            {**valid, "status": True},
            {**valid, "priority": None},
            {**valid, "created_at": "yesterday"},
            {**valid, "updated_at": []},
            {**valid, "updated_at": 0.0},
            {**valid, "created_at": float("inf")},
        ]
        for index, row in enumerate(cases):
            with self.subTest(case=index), writable_temp_directory() as tmp:
                path = Path(tmp) / "tasks.json"
                damaged = json.dumps([row]).encode()
                path.write_bytes(damaged)

                with self.assertRaisesRegex(ValueError, "schema"):
                    TaskStore(path).add("new task")

                self.assertEqual(path.read_bytes(), damaged)

    def test_task_store_concurrent_instances_preserve_all_updates(self):
        with writable_temp_directory() as tmp:
            path = Path(tmp) / "tasks.json"
            titles = [f"independent task {index}" for index in range(32)]

            def add(title):
                return TaskStore(path).add(title).id

            with ThreadPoolExecutor(max_workers=16) as executor:
                ids = list(executor.map(add, titles))

            tasks = TaskStore(path).list()
            self.assertEqual({task.id for task in tasks}, set(ids))
            self.assertEqual({task.title for task in tasks}, set(titles))

    def test_savings_ledger_persists_receipts_and_totals(self):
        with writable_temp_directory() as tmp:
            ledger = SavingsLedger(Path(tmp) / "receipts.jsonl")
            ledger.record({"input_tokens_before": 100, "input_tokens_after": 60})
            ledger.record({"input_tokens_before": 20, "input_tokens_after": 10})
            self.assertEqual(
                ledger.summary(),
                {
                    "requests": 2,
                    "tokens_before": 120,
                    "tokens_after": 70,
                    "tokens_saved": 50,
                    "reduction_pct": 41.67,
                },
            )


if __name__ == "__main__":
    unittest.main()
