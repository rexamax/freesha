"""Experimental, deterministic JOY routing preview for Freesha.

JOY means Justify, Optimize, Yield. It consumes only user-supplied fixture
estimates and never sends a provider request or initiates spend.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from freesha_core import ContentStore, EconomyContextPlanner, TokenEstimator


class JoyRouter:
    """Choose the least expensive fixture model that satisfies the task."""

    TASK_TYPES = frozenset({"coding", "extraction", "planning", "vision"})
    CAPABILITIES = frozenset({"text", "json", "code", "vision", "tool-use"})
    MAX_INPUT_COST_PER_MILLION = Decimal("1000000")
    MAX_REQUEST_INPUT_COST = Decimal("1000000000")
    MAX_CONTEXT_TOKEN_BUDGET = 10_000_000
    _NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
    _MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,95}")
    _CONTEXT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}")
    _SOURCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 .,;:()'_-]{0,159}")
    _SECRET_LIKE = re.compile(
        r"(?i)(?:sk-[A-Za-z0-9_-]{12,}|"
        r"(?:^|[-_.:/\s])(secret|password|bearer|api[-_\s]?key)(?:$|[-_.:/\s]))"
    )
    _TASK_KEYS = frozenset(
        {
            "task",
            "task_type",
            "required_capabilities",
            "required_tools",
            "minimum_quality",
            "approved_privacy_policy",
            "data_class",
            "maximum_latency_tier",
            "maximum_input_cost",
        }
    )
    _CONTEXT_ITEM_KEYS = frozenset({"id", "content", "kind", "required", "priority"})
    _CONTEXT_KINDS = frozenset({"text", "json", "log", "code"})
    _CONTEXT_PRIORITIES = frozenset({"low", "normal", "high", "critical"})
    _CATALOG_KEYS = frozenset(
        {
            "fixture_only",
            "catalog_id",
            "source_label",
            "as_of",
            "currency",
            "cost_unit",
            "latency_tiers",
            "supported_tools",
            "supported_privacy_policies",
            "supported_data_classes",
            "models",
        }
    )
    _MODEL_KEYS = frozenset(
        {
            "id",
            "task_types",
            "capabilities",
            "tools",
            "quality_by_task",
            "privacy_policies",
            "data_classes",
            "latency_tier",
            "input_cost_per_million",
        }
    )

    def __init__(self, content_store: ContentStore) -> None:
        self.store = content_store
        self.tokens = TokenEstimator()

    @staticmethod
    def _require_keys(value: dict[str, Any], required: frozenset[str], label: str) -> None:
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"{label} missing required fields: {','.join(missing)}")

    @classmethod
    def _reject_sensitive(cls, value: str, label: str) -> None:
        if cls._SECRET_LIKE.search(value):
            raise ValueError(f"{label} may be sensitive; use a public alias")

    @classmethod
    def _name_list(
        cls,
        value: Any,
        label: str,
        *,
        allowed: set[str] | frozenset[str] | None = None,
        allow_empty: bool = False,
    ) -> list[str]:
        if not isinstance(value, list) or (not value and not allow_empty):
            kind = "possibly empty" if allow_empty else "non-empty"
            raise ValueError(f"{label} must be a {kind} list")
        names = []
        for raw in value:
            if not isinstance(raw, str) or not cls._NAME.fullmatch(raw):
                raise ValueError(f"{label} contains an invalid name")
            cls._reject_sensitive(raw, label)
            names.append(raw)
        if len(names) != len(set(names)):
            raise ValueError(f"{label} must not contain duplicates")
        if allowed is not None:
            unsupported = sorted(set(names) - set(allowed))
            if unsupported:
                raise ValueError(f"{label} contains unsupported names: {','.join(unsupported)}")
        return names

    @staticmethod
    def _number(
        value: Any,
        label: str,
        *,
        minimum: Decimal,
        maximum: Decimal | None = None,
    ) -> Decimal:
        if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
            raise ValueError(f"{label} must be numeric")
        number = Decimal(str(value))
        if not number.is_finite() or number < minimum:
            raise ValueError(f"{label} is outside the supported range")
        if maximum is not None and number > maximum:
            raise ValueError(f"{label} is outside the supported range")
        return number

    @classmethod
    def _validate_catalog(cls, catalog: dict[str, Any]) -> None:
        cls._require_keys(catalog, cls._CATALOG_KEYS, "catalog")
        if catalog.get("fixture_only") is not True:
            raise ValueError("catalog fixture_only must be true for experimental JOY")
        if not isinstance(catalog["catalog_id"], str) or not cls._MODEL_ID.fullmatch(
            catalog["catalog_id"]
        ):
            raise ValueError("catalog_id must be a public-safe identifier")
        cls._reject_sensitive(catalog["catalog_id"], "catalog_id")
        if not isinstance(catalog["source_label"], str) or not cls._SOURCE.fullmatch(
            catalog["source_label"]
        ):
            raise ValueError("source_label must not contain paths, URLs, or private data")
        cls._reject_sensitive(catalog["source_label"], "source_label")
        try:
            date.fromisoformat(catalog["as_of"])
        except (TypeError, ValueError) as exc:
            raise ValueError("as_of must be an ISO date") from exc
        if not isinstance(catalog["currency"], str) or not re.fullmatch(
            r"[A-Z]{3}", catalog["currency"]
        ):
            raise ValueError("currency must be a three-letter uppercase code")
        if catalog["cost_unit"] != "per_million_input_tokens":
            raise ValueError("cost_unit must be per_million_input_tokens")

        latency_tiers = cls._name_list(catalog["latency_tiers"], "latency_tiers")
        supported_tools = cls._name_list(
            catalog["supported_tools"], "supported_tools", allow_empty=True
        )
        privacy_policies = cls._name_list(
            catalog["supported_privacy_policies"], "supported_privacy_policies"
        )
        data_classes = cls._name_list(
            catalog["supported_data_classes"], "supported_data_classes"
        )
        models = catalog["models"]
        if not isinstance(models, list) or not models:
            raise ValueError("models must be a non-empty list")

        seen_ids: set[str] = set()
        for index, candidate in enumerate(models):
            label = f"models[{index}]"
            if not isinstance(candidate, dict):
                raise ValueError(f"{label} must be an object")
            cls._require_keys(candidate, cls._MODEL_KEYS, label)
            model_id = candidate["id"]
            if not isinstance(model_id, str) or not cls._MODEL_ID.fullmatch(model_id):
                raise ValueError(f"{label}.id must be a public-safe identifier")
            if not model_id.startswith("fixture/"):
                raise ValueError(f"{label}.id must use the fixture/ namespace")
            cls._reject_sensitive(model_id, f"{label}.id")
            if model_id in seen_ids:
                raise ValueError("model ids must be unique")
            seen_ids.add(model_id)
            task_types = cls._name_list(
                candidate["task_types"], f"{label}.task_types", allowed=cls.TASK_TYPES
            )
            cls._name_list(
                candidate["capabilities"],
                f"{label}.capabilities",
                allowed=cls.CAPABILITIES,
            )
            cls._name_list(
                candidate["tools"],
                f"{label}.tools",
                allowed=set(supported_tools),
                allow_empty=True,
            )
            cls._name_list(
                candidate["privacy_policies"],
                f"{label}.privacy_policies",
                allowed=set(privacy_policies),
            )
            cls._name_list(
                candidate["data_classes"],
                f"{label}.data_classes",
                allowed=set(data_classes),
            )
            if candidate["latency_tier"] not in latency_tiers:
                raise ValueError(f"{label}.latency_tier is not declared")
            cls._number(
                candidate["input_cost_per_million"],
                f"{label}.input_cost_per_million",
                minimum=Decimal(0),
                maximum=cls.MAX_INPUT_COST_PER_MILLION,
            )
            quality = candidate["quality_by_task"]
            if not isinstance(quality, dict) or set(quality) != set(task_types):
                raise ValueError(f"{label}.quality_by_task must match task_types")
            for task_type, score in quality.items():
                cls._number(
                    score,
                    f"{label}.quality_by_task.{task_type}",
                    minimum=Decimal(0),
                    maximum=Decimal(1),
                )

    @classmethod
    def _validate_task(cls, task: dict[str, Any], catalog: dict[str, Any]) -> None:
        cls._require_keys(task, cls._TASK_KEYS, "task")
        if not isinstance(task["task"], str) or not task["task"].strip():
            raise ValueError("task.task must be a non-empty string")
        if len(task["task"]) > 100_000:
            raise ValueError("task.task is too large for JOY preview")
        if task["task_type"] not in cls.TASK_TYPES:
            raise ValueError("task_type is unsupported")
        cls._name_list(
            task["required_capabilities"],
            "required_capabilities",
            allowed=cls.CAPABILITIES,
            allow_empty=True,
        )
        cls._name_list(
            task["required_tools"],
            "required_tools",
            allowed=set(catalog["supported_tools"]),
            allow_empty=True,
        )
        cls._number(
            task["minimum_quality"],
            "minimum_quality",
            minimum=Decimal(0),
            maximum=Decimal(1),
        )
        if task["approved_privacy_policy"] not in catalog["supported_privacy_policies"]:
            raise ValueError("approved_privacy_policy is not declared by the catalog")
        if task["data_class"] not in catalog["supported_data_classes"]:
            raise ValueError("data_class is not declared by the catalog")
        if task["maximum_latency_tier"] not in catalog["latency_tiers"]:
            raise ValueError("maximum_latency_tier is not declared by the catalog")
        cls._number(
            task["maximum_input_cost"],
            "maximum_input_cost",
            minimum=Decimal(0),
            maximum=cls.MAX_REQUEST_INPUT_COST,
        )
        if "context_token_budget" in task:
            budget = task["context_token_budget"]
            if (
                isinstance(budget, bool)
                or not isinstance(budget, int)
                or not 0 < budget <= cls.MAX_CONTEXT_TOKEN_BUDGET
            ):
                raise ValueError("context_token_budget must be a positive integer")
        if "context_packet" in task:
            packet = task["context_packet"]
            if not isinstance(packet, dict) or not isinstance(packet.get("items"), list):
                raise ValueError("context_packet must be an object with an items list")
            if "task" in packet and packet["task"] != task["task"]:
                raise ValueError("context_packet.task must match task.task")
            seen_ids: set[str] = set()
            for index, item in enumerate(packet["items"]):
                if not isinstance(item, dict):
                    raise ValueError(f"context_packet.items[{index}] must be an object")
                unknown = sorted(set(item) - cls._CONTEXT_ITEM_KEYS)
                if unknown:
                    raise ValueError("context item contains unsupported fields")
                item_id = item.get("id")
                if not isinstance(item_id, str) or not cls._CONTEXT_ID.fullmatch(item_id):
                    raise ValueError("context item id must be a public-safe identifier")
                cls._reject_sensitive(item_id, "context item id")
                if item_id in seen_ids:
                    raise ValueError("context item ids must be unique")
                seen_ids.add(item_id)
                if not isinstance(item.get("content"), str):
                    raise ValueError("context item content must be a string")
                if len(item["content"]) > 10_000_000:
                    raise ValueError("context item content exceeds the preview limit")
                if "required" in item and not isinstance(item["required"], bool):
                    raise ValueError("context item required must be a boolean")
                if item.get("priority", "normal") not in cls._CONTEXT_PRIORITIES:
                    raise ValueError("context item priority is unsupported")
                if item.get("kind", "text") not in cls._CONTEXT_KINDS:
                    raise ValueError("context item kind is unsupported")

    def route(self, task: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(task, dict) or not isinstance(catalog, dict):
            raise TypeError("task and catalog must be JSON objects")
        self._validate_catalog(catalog)
        self._validate_task(task, catalog)
        task_text = str(task["task"])
        transformations: list[str] = []
        recovery_refs: list[dict[str, str]] = []
        context_details: dict[str, Any] = {}
        if "context_packet" in task:
            packet = {
                "task": task_text,
                "items": task["context_packet"]["items"],
            }
            token_budget = int(task.get("context_token_budget", 2000))
            plan = EconomyContextPlanner(self.store).build(packet, token_budget=token_budget)
            estimated_tokens = int(plan.receipt["input_tokens_after"])
            transformations.append("economy-context-planner")
            if plan.receipt["duplicates_removed"]:
                transformations.append("exact-context-deduplication")
            if plan.receipt["tokens_saved"]:
                transformations.append("recoverable-context-reduction")
            recovery_refs = [
                {"context_id": context_id, "handle": handle}
                for context_id, handle in sorted(plan.receipt["recovery_keys"].items())
            ]
            context_details = {
                "input_tokens_before": plan.receipt["input_tokens_before"],
                "token_budget": plan.receipt["token_budget"],
                "selected_ids": plan.receipt["selected_ids"],
                "omitted_ids": plan.receipt["omitted_ids"],
                "quality_gates": plan.receipt["quality_gates"],
            }
        else:
            estimated_tokens = self.tokens.count(task_text)
        optimized_context = {
            "estimated_input_tokens": estimated_tokens,
            "tokenizer": (
                "tiktoken/o200k_base-local-estimate"
                if self.tokens.encoder
                else "offline-byte-estimate"
            ),
            "transformations": transformations,
            "recovery_refs": recovery_refs,
            **context_details,
        }
        latency_order = {
            tier: index for index, tier in enumerate(catalog["latency_tiers"])
        }
        candidates: list[tuple[Decimal, int, str, Decimal]] = []
        rejected: list[dict[str, Any]] = []
        for candidate in catalog["models"]:
            supports_task = task["task_type"] in candidate["task_types"]
            quality = Decimal(
                str(candidate["quality_by_task"].get(task["task_type"], 0))
            )
            cost = (
                Decimal(estimated_tokens)
                * Decimal(str(candidate["input_cost_per_million"]))
                / Decimal(1_000_000)
            )
            reasons = []
            if not supports_task:
                reasons.append("task_type_not_supported")
            elif quality < Decimal(str(task["minimum_quality"])):
                reasons.append("quality_below_minimum")
            missing_capabilities = sorted(
                set(task["required_capabilities"]) - set(candidate["capabilities"])
            )
            if missing_capabilities:
                reasons.append(f"missing_capabilities:{','.join(missing_capabilities)}")
            missing_tools = sorted(set(task["required_tools"]) - set(candidate["tools"]))
            if missing_tools:
                reasons.append(f"missing_tools:{','.join(missing_tools)}")
            if task["approved_privacy_policy"] not in candidate["privacy_policies"]:
                reasons.append("privacy_policy_not_approved")
            if task["data_class"] not in candidate["data_classes"]:
                reasons.append("data_class_not_supported")
            if latency_order[candidate["latency_tier"]] > latency_order[
                task["maximum_latency_tier"]
            ]:
                reasons.append("latency_exceeds_ceiling")
            if cost > Decimal(str(task["maximum_input_cost"])):
                reasons.append("projected_input_cost_exceeds_budget")
            if reasons:
                rejected.append(
                    {
                        "model_id": candidate["id"],
                        "projected_input_cost": float(cost),
                        "currency": catalog["currency"],
                        "reasons": reasons,
                    }
                )
                continue
            candidates.append(
                (
                    cost,
                    latency_order[candidate["latency_tier"]],
                    candidate["id"],
                    quality,
                )
            )

        fingerprint = hashlib.sha256(
            json.dumps(
                task,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        constraint_keys = (
            "task_type",
            "required_capabilities",
            "required_tools",
            "minimum_quality",
            "approved_privacy_policy",
            "data_class",
            "maximum_latency_tier",
            "maximum_input_cost",
        )
        constraint_summary = {key: task[key] for key in constraint_keys}
        if "context_token_budget" in task:
            constraint_summary["context_token_budget"] = task["context_token_budget"]
        common_receipt = {
            "mode": "joy-dry-run",
            "experimental": True,
            "dry_run": True,
            "task_fingerprint": f"sha256:{fingerprint}",
            "rejected_candidates": rejected,
            "optimized_context": optimized_context,
            "constraint_summary": constraint_summary,
            "escalation_policy": "No automatic escalation; human review is required.",
            "catalog": {
                "id": catalog["catalog_id"],
                "as_of": catalog["as_of"],
                "source": catalog["source_label"],
                "fixture_only": catalog["fixture_only"],
            },
            "disclaimers": [
                "Catalog quality, latency, and pricing values are user-supplied fixture estimates.",
                "This preview does not claim live model availability or provider pricing.",
            ],
            "network_calls": 0,
            "provider_request_sent": False,
            "automatic_spend": False,
        }
        if not candidates:
            return {
                **common_receipt,
                "route_found": False,
                "failure_reason": "no_model_satisfies_all_constraints",
                "selected_model": None,
                "selection_reasons": [],
                "fallbacks": [],
            }

        ranked_candidates = sorted(candidates)
        selected_cost, _, selected_id, selected_quality = ranked_candidates[0]
        fallbacks = [
            {
                "model_id": candidate_id,
                "quality_score": float(quality),
                "quality_headroom": float(quality - selected_quality),
                "projected_input_cost": float(cost),
                "currency": catalog["currency"],
            }
            for cost, _, candidate_id, quality in ranked_candidates[1:]
            if quality > selected_quality
        ]
        return {
            **common_receipt,
            "route_found": True,
            "selected_model": {
                "id": selected_id,
                "quality_score": float(selected_quality),
                "estimated_input_tokens": estimated_tokens,
                "projected_input_cost": float(selected_cost),
                "currency": catalog["currency"],
            },
            "selection_reasons": [
                "satisfies the supplied constraints",
                "lowest projected input cost among eligible fixture models",
            ],
            "fallbacks": fallbacks,
        }


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object for the CLI without retaining its path in receipts."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value
