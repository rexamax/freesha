import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from freesha_core import ContentStore, EconomyContextPlanner, main
from freesha_joy import JoyRouter


def catalog(models):
    return {
        "fixture_only": True,
        "catalog_id": "joy-synthetic-v1",
        "source_label": "Synthetic fixture estimates, not provider pricing",
        "as_of": "2026-07-19",
        "currency": "USD",
        "cost_unit": "per_million_input_tokens",
        "latency_tiers": ["fast", "standard", "slow"],
        "supported_tools": ["json-output", "repository-search"],
        "supported_privacy_policies": ["public", "no-training"],
        "supported_data_classes": ["synthetic-public", "internal"],
        "models": models,
    }


def model(model_id, *, cost, quality=0.9, latency="fast"):
    return {
        "id": model_id,
        "task_types": ["extraction"],
        "capabilities": ["text", "json"],
        "tools": ["json-output"],
        "quality_by_task": {"extraction": quality},
        "privacy_policies": ["public"],
        "data_classes": ["synthetic-public"],
        "latency_tier": latency,
        "input_cost_per_million": cost,
    }


def task():
    return {
        "task": "Extract the synthetic ticket identifier as JSON.",
        "task_type": "extraction",
        "required_capabilities": ["text", "json"],
        "required_tools": ["json-output"],
        "minimum_quality": 0.85,
        "approved_privacy_policy": "public",
        "data_class": "synthetic-public",
        "maximum_latency_tier": "standard",
        "maximum_input_cost": 0.01,
    }


class JoyRouterTests(unittest.TestCase):
    def test_selects_cheapest_eligible_model(self):
        receipt = JoyRouter(ContentStore(Path(".test-state/joy-cheapest"))).route(
            task(),
            catalog(
                [
                    model("fixture/expensive", cost=3.0),
                    model("fixture/economy", cost=0.2),
                ]
            ),
        )

        self.assertTrue(receipt["route_found"])
        self.assertEqual(receipt["selected_model"]["id"], "fixture/economy")
        self.assertTrue(receipt["dry_run"])
        self.assertEqual(receipt["network_calls"], 0)
        self.assertFalse(receipt["provider_request_sent"])
        self.assertFalse(receipt["automatic_spend"])

    def test_hard_filters_quality_and_required_capabilities(self):
        ineligible = model("fixture/cheap-but-ineligible", cost=0.01, quality=0.8)
        ineligible["capabilities"] = ["text"]
        receipt = JoyRouter(ContentStore(Path(".test-state/joy-filters"))).route(
            task(),
            catalog([ineligible, model("fixture/qualified", cost=1.0)]),
        )

        self.assertEqual(receipt["selected_model"]["id"], "fixture/qualified")
        rejected = {item["model_id"]: item["reasons"] for item in receipt["rejected_candidates"]}
        self.assertIn("quality_below_minimum", rejected["fixture/cheap-but-ineligible"])
        self.assertIn(
            "missing_capabilities:json", rejected["fixture/cheap-but-ineligible"]
        )

    def test_budget_privacy_data_class_tools_and_latency_are_hard_limits(self):
        privacy = model("fixture/privacy-mismatch", cost=0.01)
        privacy["privacy_policies"] = ["no-training"]
        privacy["data_classes"] = ["internal"]
        missing_tool = model("fixture/tool-mismatch", cost=0.02)
        missing_tool["tools"] = []
        slow = model("fixture/too-slow", cost=0.03, latency="slow")
        over_budget = model("fixture/over-budget", cost=1_000_000)
        receipt = JoyRouter(ContentStore(Path(".test-state/joy-limits"))).route(
            task(),
            catalog(
                [
                    privacy,
                    missing_tool,
                    slow,
                    over_budget,
                    model("fixture/eligible", cost=1.0),
                ]
            ),
        )

        self.assertEqual(receipt["selected_model"]["id"], "fixture/eligible")
        rejected = {item["model_id"]: item["reasons"] for item in receipt["rejected_candidates"]}
        self.assertIn("privacy_policy_not_approved", rejected["fixture/privacy-mismatch"])
        self.assertIn("data_class_not_supported", rejected["fixture/privacy-mismatch"])
        self.assertIn("missing_tools:json-output", rejected["fixture/tool-mismatch"])
        self.assertIn("latency_exceeds_ceiling", rejected["fixture/too-slow"])
        self.assertIn("projected_input_cost_exceeds_budget", rejected["fixture/over-budget"])
        self.assertTrue(
            all(
                "projected_input_cost" in item and item["currency"] == "USD"
                for item in receipt["rejected_candidates"]
            )
        )

    def test_equal_cost_uses_latency_then_model_id_as_deterministic_ties(self):
        receipt = JoyRouter(ContentStore(Path(".test-state/joy-tie"))).route(
            task(),
            catalog(
                [
                    model("fixture/a-slow", cost=1.0, latency="standard"),
                    model("fixture/z-fast", cost=1.0, latency="fast"),
                    model("fixture/y-fast", cost=1.0, latency="fast"),
                ]
            ),
        )

        self.assertEqual(receipt["selected_model"]["id"], "fixture/y-fast")

    def test_fallbacks_are_eligible_ordered_and_have_greater_quality_headroom(self):
        receipt = JoyRouter(ContentStore(Path(".test-state/joy-fallbacks"))).route(
            task(),
            catalog(
                [
                    model("fixture/selected", cost=0.2, quality=0.9),
                    model("fixture/same-quality", cost=0.3, quality=0.9),
                    model("fixture/high-b", cost=2.0, quality=0.96, latency="standard"),
                    model("fixture/high-a", cost=1.0, quality=0.95),
                ]
            ),
        )

        self.assertEqual(
            [item["model_id"] for item in receipt["fallbacks"]],
            ["fixture/high-a", "fixture/high-b"],
        )
        self.assertTrue(
            all(item["quality_score"] > 0.9 for item in receipt["fallbacks"])
        )

    def test_no_eligible_model_fails_closed_with_reasons_without_task_text(self):
        private_task = task()
        private_task["task"] = "PRIVATE-STRING-DO-NOT-ECHO"
        low_quality = model("fixture/low", cost=0.1, quality=0.2)
        no_json = model("fixture/no-json", cost=0.1)
        no_json["capabilities"] = ["text"]

        receipt = JoyRouter(ContentStore(Path(".test-state/joy-none"))).route(
            private_task,
            catalog([low_quality, no_json]),
        )

        self.assertFalse(receipt["route_found"])
        self.assertIsNone(receipt["selected_model"])
        self.assertEqual(len(receipt["rejected_candidates"]), 2)
        self.assertEqual(receipt["failure_reason"], "no_model_satisfies_all_constraints")
        self.assertNotIn("PRIVATE-STRING-DO-NOT-ECHO", json.dumps(receipt))
        self.assertEqual(receipt["network_calls"], 0)
        self.assertFalse(receipt["automatic_spend"])

    def test_rejects_catalog_that_is_not_explicitly_fixture_only(self):
        live_catalog = catalog([model("fixture/one", cost=1.0)])
        live_catalog["fixture_only"] = False

        with self.assertRaisesRegex(ValueError, "fixture_only must be true"):
            JoyRouter(ContentStore(Path(".test-state/joy-fixture"))).route(
                task(), live_catalog
            )

    def test_rejects_malformed_duplicate_and_unsupported_inputs(self):
        duplicate_catalog = catalog(
            [model("fixture/duplicate", cost=1.0), model("fixture/duplicate", cost=2.0)]
        )
        negative_cost_catalog = catalog([model("fixture/negative", cost=-1.0)])
        unknown_capability_catalog = catalog([model("fixture/unknown-cap", cost=1.0)])
        unknown_capability_catalog["models"][0]["capabilities"].append("telepathy")
        bad_source_catalog = catalog([model("fixture/source", cost=1.0)])
        bad_source_catalog["source_label"] = "C:\\private\\catalog.json"
        bad_task_type = task()
        bad_task_type["task_type"] = "unsupported-chat"
        bad_quality = task()
        bad_quality["minimum_quality"] = 1.1
        unknown_tool = task()
        unknown_tool["required_tools"] = ["undeclared-tool"]

        cases = [
            ({}, catalog([model("fixture/one", cost=1.0)])),
            (task(), {"fixture_only": True}),
            (task(), catalog([])),
            (task(), duplicate_catalog),
            (task(), negative_cost_catalog),
            (task(), unknown_capability_catalog),
            (task(), bad_source_catalog),
            (bad_task_type, catalog([model("fixture/one", cost=1.0)])),
            (bad_quality, catalog([model("fixture/one", cost=1.0)])),
            (unknown_tool, catalog([model("fixture/one", cost=1.0)])),
        ]
        router = JoyRouter(ContentStore(Path(".test-state/joy-invalid")))

        for task_value, catalog_value in cases:
            with (
                self.subTest(task=task_value.get("task_type"), catalog=catalog_value),
                self.assertRaises(ValueError),
            ):
                router.route(task_value, catalog_value)

    def test_context_packet_reuses_economy_planner_and_exposes_recovery_handles(self):
        raw_json = json.dumps(
            {"tickets": [{"id": f"SYN-{index:03d}", "status": "open"} for index in range(30)]},
            indent=4,
        )
        packet = {
            "items": [
                {
                    "id": "fixture-json",
                    "kind": "json",
                    "content": raw_json,
                    "required": True,
                },
                {"id": "duplicate-json", "kind": "json", "content": raw_json},
            ]
        }
        routed_task = task()
        routed_task["context_packet"] = packet
        routed_task["context_token_budget"] = 250
        store = ContentStore(Path(".test-state/joy-planner"))

        receipt = JoyRouter(store).route(
            routed_task, catalog([model("fixture/economy", cost=0.2)])
        )
        direct_plan = EconomyContextPlanner(store).build(
            {"task": routed_task["task"], "items": packet["items"]}, token_budget=250
        )

        optimized = receipt["optimized_context"]
        self.assertEqual(
            optimized["estimated_input_tokens"], direct_plan.receipt["input_tokens_after"]
        )
        self.assertIn("economy-context-planner", optimized["transformations"])
        self.assertTrue(optimized["recovery_refs"])
        fixture_ref = next(
            item for item in optimized["recovery_refs"] if item["context_id"] == "fixture-json"
        )
        self.assertEqual(store.get(fixture_ref["handle"]), raw_json)

    def test_receipt_omits_raw_private_context_secrets_and_absolute_paths(self):
        private_task = task()
        private_task["task"] = "Review PRIVATE-CREDENTIAL-DO-NOT-SERIALIZE"
        private_task["context_packet"] = {
            "items": [
                {
                    "id": "private-fixture",
                    "content": "C:\\Users\\alice\\private.txt PRIVATE-CREDENTIAL-RAW-CONTEXT",
                    "required": True,
                }
            ]
        }
        receipt = JoyRouter(ContentStore(Path(".test-state/joy-private"))).route(
            private_task,
            catalog([model("fixture/economy", cost=0.2)]),
        )

        serialized = json.dumps(receipt)
        self.assertNotIn("DO-NOT-SERIALIZE", serialized)
        self.assertNotIn("RAW-CONTEXT", serialized)
        self.assertNotIn("C:\\\\Users", serialized)
        self.assertNotIn(str(Path.cwd().resolve()), serialized)

    def test_secret_like_context_id_is_rejected_before_it_can_enter_receipt(self):
        private_task = task()
        private_task["context_packet"] = {
            "items": [
                {
                    "id": "sk-proj-DO-NOT-ECHO",
                    "content": "synthetic public content",
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "public alias"):
            JoyRouter(ContentStore(Path(".test-state/joy-secret-id"))).route(
                private_task,
                catalog([model("fixture/economy", cost=0.2)]),
            )

    def test_secret_like_catalog_metadata_is_rejected_before_receipt(self):
        secret_source = catalog([model("fixture/economy", cost=0.2)])
        secret_source["source_label"] = "sk-proj-DO-NOT-SERIALIZE"
        secret_model = catalog(
            [model("fixture/sk-proj-DO-NOT-SERIALIZE", cost=0.2)]
        )
        secret_tool = catalog([model("fixture/economy", cost=0.2)])
        secret_tool["supported_tools"] = ["sk-proj-DO-NOT-SERIALIZE"]
        secret_tool["models"][0]["tools"] = ["sk-proj-DO-NOT-SERIALIZE"]
        router = JoyRouter(ContentStore(Path(".test-state/joy-secret-metadata")))

        for catalog_value in (secret_source, secret_model, secret_tool):
            with self.assertRaisesRegex(ValueError, "sensitive"):
                router.route(task(), catalog_value)

    def test_context_items_require_exact_supported_types(self):
        invalid_items = [
            {"id": "bad-content", "content": 42},
            {"id": "bad-required", "content": "safe", "required": "false"},
            {"id": "bad-priority", "content": "safe", "priority": "urgent"},
            {"id": "bad-kind", "content": "safe", "kind": "binary"},
        ]
        router = JoyRouter(ContentStore(Path(".test-state/joy-item-types")))

        for invalid_item in invalid_items:
            invalid_task = task()
            invalid_task["context_packet"] = {"items": [invalid_item]}
            with self.subTest(item_id=invalid_item["id"]), self.assertRaisesRegex(
                ValueError, "context item"
            ):
                router.route(
                    invalid_task,
                    catalog([model("fixture/economy", cost=0.2)]),
                )

    def test_rejects_cost_numbers_too_large_for_strict_json_receipts(self):
        huge_cost_catalog = catalog(
            [model("fixture/overflow", cost=10**1000)]
        )
        huge_budget_task = task()
        huge_budget_task["maximum_input_cost"] = 10**1000
        router = JoyRouter(ContentStore(Path(".test-state/joy-cost-range")))

        for task_value, catalog_value in (
            (task(), huge_cost_catalog),
            (huge_budget_task, catalog([model("fixture/economy", cost=0.2)])),
        ):
            with self.assertRaisesRegex(ValueError, "supported range"):
                router.route(task_value, catalog_value)

    def test_cli_requires_explicit_dry_run_and_fails_closed(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main(["joy", "task.json", "--models", "models.json"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("JOY requires --dry-run", stderr.getvalue())

    def test_cli_dry_run_makes_no_network_request_or_spend(self):
        root = Path(".test-state/joy-cli")
        root.mkdir(parents=True, exist_ok=True)
        task_path = root / "task.json"
        catalog_path = root / "models.json"
        task_path.write_text(json.dumps(task()), encoding="utf-8")
        catalog_path.write_text(
            json.dumps(catalog([model("fixture/economy", cost=0.2)])), encoding="utf-8"
        )
        stdout = io.StringIO()

        with (
            patch("urllib.request.urlopen", side_effect=AssertionError("network attempted")),
            redirect_stdout(stdout),
        ):
            exit_code = main(
                [
                    "joy",
                    str(task_path),
                    "--models",
                    str(catalog_path),
                    "--dry-run",
                    "--store",
                    str(root / "blobs"),
                ]
            )

        receipt = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(receipt["catalog"]["fixture_only"])
        self.assertEqual(receipt["network_calls"], 0)
        self.assertFalse(receipt["provider_request_sent"])
        self.assertFalse(receipt["automatic_spend"])

    def test_cli_reports_duplicate_catalog_without_traceback(self):
        root = Path(".test-state/joy-cli-invalid")
        root.mkdir(parents=True, exist_ok=True)
        task_path = root / "task.json"
        catalog_path = root / "models.json"
        task_path.write_text(json.dumps(task()), encoding="utf-8")
        duplicate = model("fixture/duplicate", cost=0.2)
        catalog_path.write_text(
            json.dumps(catalog([duplicate, duplicate])), encoding="utf-8"
        )
        stderr = io.StringIO()

        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main(
                [
                    "joy",
                    str(task_path),
                    "--models",
                    str(catalog_path),
                    "--dry-run",
                ]
            )

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("model ids must be unique", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
