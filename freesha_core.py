"""Freesha: local-first context optimization for OpenAI-compatible workflows.

The core stays deterministic and provider-agnostic. It never sends data anywhere
unless the caller explicitly invokes ``forward_chat``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import tiktoken  # Optional local approximation; provider usage remains authoritative.
except ImportError:  # pragma: no cover - exercised when optional dependency absent
    tiktoken = None


@dataclass(frozen=True)
class OptimizationResult:
    optimized: str
    original_tokens: int
    optimized_tokens: int
    bytes_saved: int
    mode: str

    @property
    def tokens_saved(self) -> int:
        return max(0, self.original_tokens - self.optimized_tokens)

    @property
    def reduction_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return round(self.tokens_saved / self.original_tokens * 100, 2)


@dataclass(frozen=True)
class PayloadResult:
    payload: dict[str, Any]
    receipt: dict[str, Any]


@dataclass(frozen=True)
class CacheResult:
    key: str
    hit: bool
    tokens: int


@dataclass(frozen=True)
class CompactionResult:
    output: str
    original_tokens: int
    compacted_tokens: int
    changed: bool
    recovery_key: str | None
    mode: str

    @property
    def tokens_saved(self) -> int:
        return max(0, self.original_tokens - self.compacted_tokens)

    @property
    def reduction_pct(self) -> float:
        if not self.original_tokens:
            return 0.0
        return round(self.tokens_saved / self.original_tokens * 100, 2)


@dataclass(frozen=True)
class ContextPlanResult:
    context: str
    receipt: dict[str, Any]


@dataclass
class Task:
    id: str
    title: str
    status: str = "todo"
    priority: str = "normal"
    created_at: float = 0.0
    updated_at: float = 0.0


class TokenEstimator:
    def __init__(self) -> None:
        self.encoder = None
        if tiktoken is not None:
            try:
                self.encoder = tiktoken.get_encoding("o200k_base")
            except Exception:
                self.encoder = None

    def count(self, value: Any) -> int:
        text = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        )
        if self.encoder is not None:
            return len(self.encoder.encode(text))
        # Conservative offline estimate. Receipt labels this as estimated.
        return max(0, (len(text.encode("utf-8")) + 3) // 4)


class FreeshaOptimizer:
    """Deterministic transformations with a receipt for every optimization."""

    def __init__(self) -> None:
        self.tokens = TokenEstimator()

    def optimize_json(self, text: str) -> OptimizationResult:
        source = str(text)
        try:
            parsed = json.loads(source)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("optimize_json expects valid JSON text") from exc
        optimized = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return OptimizationResult(
            optimized=optimized,
            original_tokens=self.tokens.count(source),
            optimized_tokens=self.tokens.count(optimized),
            bytes_saved=max(0, len(source.encode()) - len(optimized.encode())),
            mode="json-minify",
        )

    def _maybe_minify_json_string(self, value: str) -> tuple[str, bool]:
        stripped = value.strip()
        if not stripped or stripped[0] not in "[{":
            return value, False
        try:
            result = self.optimize_json(value)
        except ValueError:
            return value, False
        return result.optimized, result.optimized != value

    def optimize_payload(self, payload: dict[str, Any]) -> PayloadResult:
        """Optimize only JSON-looking message content; preserve other text exactly."""
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dictionary")
        optimized = json.loads(json.dumps(payload, ensure_ascii=False))
        transformations: list[str] = []
        before = self.tokens.count(payload)

        messages = optimized.get("messages", [])
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    compact, changed = self._maybe_minify_json_string(content)
                    if changed:
                        message["content"] = compact
                        transformations.append("json-message-minify")

        after = self.tokens.count(optimized)
        receipt = {
            "input_tokens_before": before,
            "input_tokens_after": after,
            "estimated_input_reduction_pct": round(max(0, before - after) / before * 100, 2)
            if before
            else 0.0,
            "tokenizer": "tiktoken/o200k_base-local-estimate"
            if self.tokens.encoder
            else "offline-byte-estimate",
            "transformations": transformations,
            "semantic_policy": "lossless-for-minified-json; other text unchanged",
        }
        return PayloadResult(payload=optimized, receipt=receipt)

    def python_skeleton(self, source: str) -> str:
        """Return a compact structural view while retaining callable signatures."""
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise ValueError(f"invalid Python source: {exc}") from exc

        lines: list[str] = []

        def signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            args = ast.unparse(node.args)
            result = f"{prefix} {node.name}({args})"
            if node.returns is not None:
                result += f" -> {ast.unparse(node.returns)}"
            return result + ":"

        def visit(body: Iterable[ast.stmt], indent: int = 0) -> None:
            pad = " " * indent
            for node in body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    lines.append(pad + signature(node))
                elif isinstance(node, ast.ClassDef):
                    bases = ""
                    if node.bases:
                        bases = "(" + ", ".join(ast.unparse(x) for x in node.bases) + ")"
                    lines.append(pad + f"class {node.name}{bases}:")
                    visit(node.body, indent + 4)
                elif isinstance(
                    node,
                    (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith),
                ):
                    # Show control-flow headers but do not recursively dump implementation.
                    segment = ast.get_source_segment(source, node) or type(node).__name__
                    header = segment.splitlines()[0].strip()
                    lines.append(pad + header + " …")

        visit(tree.body)
        return "\n".join(lines)

    def compact_system_prompt(self, level: str = "full") -> str:
        levels = {
            "lite": "Be concise. Skip pleasantries. Preserve code, commands, errors.",
            "full": (
                "Answer directly. No pleasantries or repeated context. "
                "Preserve code, commands, errors exactly."
            ),
            "ultra": "Direct answer only. Preserve code, commands, errors exactly.",
        }
        if level not in levels:
            raise ValueError("level must be lite, full, or ultra")
        return levels[level]

    def optimize_request(
        self, payload: dict[str, Any], output_mode: str | None = None
    ) -> PayloadResult:
        """Apply lossless transforms without adding non-standard API fields.

        ``output_mode`` remains accepted for backwards compatibility, but it is
        receipt-only. Callers that want shorter answers should set a provider's
        documented output-token limit explicitly.
        """
        result = self.optimize_payload(payload)
        if output_mode:
            result.receipt["requested_output_mode"] = output_mode
        return result

    def prepare_openai_cache(
        self,
        payload: dict[str, Any],
        cache_key: str,
        *,
        ttl: str = "30m",
    ) -> PayloadResult:
        """Opt in to documented GPT-5.6 prompt-cache routing fields.

        This method deliberately does not guess cache boundaries or reorder
        messages. Static instructions/tools must already precede variable input.
        Cache effectiveness is verified from provider usage, not estimated here.
        """
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dictionary")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:._-]{0,127}", str(cache_key)):
            raise ValueError("cache_key must be a non-secret stable identifier")
        if ttl != "30m":
            raise ValueError("OpenAI currently supports only a 30m cache TTL")
        model = str(payload.get("model", ""))
        if not model.startswith("gpt-5.6"):
            raise ValueError("explicit cache options require a GPT-5.6 family model")

        optimized = self.optimize_payload(payload)
        optimized.payload["prompt_cache_key"] = cache_key
        optimized.payload["prompt_cache_options"] = {
            "mode": "implicit",
            "ttl": ttl,
        }
        optimized.receipt["transformations"].append("openai-prompt-cache")
        optimized.receipt["cache_policy"] = {
            "key": cache_key,
            "ttl": ttl,
            "measurement": "read cached_tokens and cache_write_tokens from provider usage",
        }
        return optimized


def _atomic_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


class ContentStore:
    """Local content-addressed blob store used for exact recovery."""

    _KEY = re.compile(r"[0-9a-f]{64}")

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._write_lock = threading.Lock()

    def put(self, content: str) -> str:
        text = str(content)
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._write_lock:
            self.path.mkdir(parents=True, exist_ok=True)
            os.chmod(self.path, 0o700)
            destination = self.path / f"{key}.txt"
            if not destination.exists():
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=self.path,
                    prefix=f".{key}.",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    handle.write(text.encode("utf-8"))
                    temp = Path(handle.name)
                try:
                    os.chmod(temp, 0o600)
                    temp.replace(destination)
                finally:
                    if temp.exists():
                        temp.unlink()
            os.chmod(destination, 0o600)
        return key

    def get(self, key: str | None) -> str:
        if key is None or not self._KEY.fullmatch(str(key)):
            raise ValueError("invalid recovery key")
        try:
            return (self.path / f"{key}.txt").read_bytes().decode("utf-8")
        except FileNotFoundError as exc:
            raise KeyError(f"recovery content not found: {key}") from exc


class ToolOutputCompactor:
    """Compact repetitive logs while keeping critical lines and exact recovery."""

    _CRITICAL = re.compile(
        r"(?i)(error|exception|traceback|fatal|failed|failure|panic|denied|timeout|warning|warn)"
    )
    _TIMESTAMP = re.compile(
        r"\b(?:\d{4}-\d{2}-\d{2}[T ][0-9:.+-]+Z?|\d{2}:\d{2}:\d{2}(?:\.\d+)?)\b"
    )
    _UUID = re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    )
    _HEX = re.compile(r"\b(?:0x)?[0-9a-fA-F]{12,64}\b")
    _KEYED_ID = re.compile(
        r"\b(job|pid|task_id|request_id|trace_id|span_id|run_id)=([^\s,]+)",
        re.IGNORECASE,
    )
    _DURATION = re.compile(r"\b\d+(?:\.\d+)?(?:ms|s|sec|seconds|us|µs)\b", re.IGNORECASE)

    def __init__(self, store: ContentStore, minimum_tokens: int = 24) -> None:
        self.store = store
        self.minimum_tokens = minimum_tokens
        self.tokens = TokenEstimator()

    def _template(self, line: str) -> str:
        value = self._TIMESTAMP.sub("<timestamp>", line)
        value = self._UUID.sub("<uuid>", value)
        value = self._HEX.sub("<hex>", value)
        value = self._KEYED_ID.sub(lambda match: f"{match.group(1)}=<id>", value)
        value = self._DURATION.sub("<duration>", value)
        return value

    def compact(self, output: str) -> CompactionResult:
        source = str(output)
        original_tokens = self.tokens.count(source)
        if original_tokens < self.minimum_tokens or len(source.splitlines()) < 3:
            return CompactionResult(
                output=source,
                original_tokens=original_tokens,
                compacted_tokens=original_tokens,
                changed=False,
                recovery_key=None,
                mode="passthrough",
            )

        runs: list[dict[str, Any]] = []
        for position, line in enumerate(source.splitlines()):
            if self._CRITICAL.search(line):
                key = f"critical:{position}"
            else:
                key = "template:" + self._template(line)
            if runs and runs[-1]["key"] == key:
                runs[-1]["count"] += 1
            else:
                runs.append({"key": key, "sample": line, "count": 1})

        compacted_lines: list[str] = []
        repeated_groups = 0
        for run in runs:
            compacted_lines.append(str(run["sample"]))
            if int(run["count"]) > 1:
                repeated_groups += 1
                compacted_lines.append(
                    f"[repeated x{run['count']}; variable timestamps/ids/durations omitted]"
                )

        if repeated_groups == 0:
            return CompactionResult(
                output=source,
                original_tokens=original_tokens,
                compacted_tokens=original_tokens,
                changed=False,
                recovery_key=None,
                mode="passthrough",
            )

        recovery_key = hashlib.sha256(source.encode("utf-8")).hexdigest()
        compacted_lines.append(f"[recover exact output with key {recovery_key}]")
        compacted = "\n".join(compacted_lines)
        compacted_tokens = self.tokens.count(compacted)
        if compacted_tokens >= original_tokens:
            return CompactionResult(
                output=source,
                original_tokens=original_tokens,
                compacted_tokens=original_tokens,
                changed=False,
                recovery_key=None,
                mode="net-loss-passthrough",
            )
        stored_key = self.store.put(source)
        if stored_key != recovery_key:  # Defensive invariant for the content-addressed store.
            raise RuntimeError("recovery key mismatch")
        return CompactionResult(
            output=compacted,
            original_tokens=original_tokens,
            compacted_tokens=compacted_tokens,
            changed=True,
            recovery_key=recovery_key,
            mode="recoverable-log-template",
        )


class EconomyContextPlanner:
    """Build a deterministic, budget-aware context packet without model calls."""

    _ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}")
    _WORDS = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]+")
    _CRITICAL = ToolOutputCompactor._CRITICAL
    _PRIORITY = {"low": 0, "normal": 2, "high": 8, "critical": 20}

    def __init__(self, content_store: ContentStore) -> None:
        self.store = content_store
        self.tokens = TokenEstimator()
        self.compactor = ToolOutputCompactor(content_store)
        self.optimizer = FreeshaOptimizer()

    @staticmethod
    def _dedupe_key(content: str) -> str:
        return str(content)

    def _terms(self, text: str) -> set[str]:
        return {word.lower() for word in self._WORDS.findall(text)}

    def _prepare_content(self, item: dict[str, Any]) -> tuple[str, str | None]:
        content = str(item.get("content", ""))
        kind = str(item.get("kind", "text"))
        if kind == "log":
            result = self.compactor.compact(content)
            return result.output, result.recovery_key
        stripped = content.strip()
        if kind == "json" or (stripped.startswith(("{", "[")) and stripped):
            try:
                result = self.optimizer.optimize_json(content)
                if result.optimized_tokens >= result.original_tokens:
                    return content, None
                recovery_key = self.store.put(content)
                return result.optimized, recovery_key
            except ValueError:
                pass
        return content, None

    def build(self, packet: dict[str, Any], token_budget: int = 2000) -> ContextPlanResult:
        if not isinstance(packet, dict):
            raise TypeError("packet must be a dictionary")
        if token_budget <= 0:
            raise ValueError("token_budget must be positive")
        task = str(packet.get("task", "")).strip()
        raw_items = packet.get("items", [])
        if not task or not isinstance(raw_items, list):
            raise ValueError("packet requires a task and an items list")

        task_terms = self._terms(task)
        seen: dict[str, dict[str, Any]] = {}
        seen_ids: set[str] = set()
        unique: list[dict[str, Any]] = []
        duplicates: list[str] = []
        recovery: dict[str, str] = {}

        for index, raw in enumerate(raw_items):
            if not isinstance(raw, dict):
                raise ValueError("each item must be a dictionary")
            item_id = str(raw.get("id", ""))
            if not self._ID.fullmatch(item_id):
                raise ValueError("item id must be a short public-safe identifier")
            if item_id in seen_ids:
                raise ValueError("item ids must be unique")
            seen_ids.add(item_id)
            original = str(raw.get("content", ""))
            dedupe_key = self._dedupe_key(original)
            content, key = self._prepare_content(raw)
            overlap = len(task_terms & self._terms(content))
            required = bool(raw.get("required", False))
            priority = self._PRIORITY.get(str(raw.get("priority", "normal")), 2)
            score = overlap * 10 + priority
            if self._CRITICAL.search(content):
                score += 50
            if required:
                score += 10_000
            candidate = {
                "id": item_id,
                "content": content,
                "original": original,
                "required": required,
                "score": score,
                "index": index,
            }

            existing = seen.get(dedupe_key)
            if existing is not None:
                if required and existing["required"]:
                    if key:
                        recovery[item_id] = key
                    unique.append(candidate)
                elif required:
                    replaced_id = str(existing["id"])
                    duplicates.append(replaced_id)
                    recovery[replaced_id] = self.store.put(str(existing["original"]))
                    existing.clear()
                    existing.update(candidate)
                    if key:
                        recovery[item_id] = key
                else:
                    duplicates.append(item_id)
                    recovery[item_id] = self.store.put(original)
                    existing["score"] = max(int(existing["score"]), score)
                continue

            if key:
                recovery[item_id] = key
            unique.append(candidate)
            seen[dedupe_key] = candidate

        ranked = sorted(
            unique,
            key=lambda item: (not bool(item["required"]), -item["score"], item["index"]),
        )
        selected: list[dict[str, Any]] = []
        prefix = f"TASK\n{task}\nCONTEXT\n"
        for item in ranked:
            candidate_items = sorted([*selected, item], key=lambda entry: entry["index"])
            candidate_blocks = [f"[{entry['id']}]\n{entry['content']}" for entry in candidate_items]
            candidate_tokens = self.tokens.count(prefix + "\n\n".join(candidate_blocks))
            if item["required"] or candidate_tokens <= token_budget:
                selected.append(item)
            else:
                recovery[item["id"]] = self.store.put(item["original"])

        selected.sort(key=lambda item: item["index"])
        blocks = [f"[{item['id']}]\n{item['content']}" for item in selected]
        context = prefix + "\n\n".join(blocks)
        before_text = f"TASK\n{task}\nCONTEXT\n" + "\n\n".join(
            f"[{item.get('id', '')}]\n{item.get('content', '')}"
            for item in raw_items
            if isinstance(item, dict)
        )
        before = self.tokens.count(before_text)
        after = self.tokens.count(context)
        required_blocks = [
            f"[{item['id']}]\n{item['content']}"
            for item in sorted(unique, key=lambda entry: entry["index"])
            if item["required"]
        ]
        mandatory_tokens = self.tokens.count(prefix + "\n\n".join(required_blocks))
        mandatory_overflow = mandatory_tokens > token_budget
        selected_ids = [str(item["id"]) for item in selected]
        omitted_ids = [str(item["id"]) for item in unique if str(item["id"]) not in selected_ids]
        receipt = {
            "input_tokens_before": before,
            "input_tokens_after": after,
            "token_budget": token_budget,
            "budget_exceeded": after > token_budget,
            "mandatory_tokens": mandatory_tokens,
            "tokens_saved": max(0, before - after),
            "reduction_pct": round(max(0, before - after) / before * 100, 2) if before else 0.0,
            "tokenizer": "tiktoken/o200k_base-local-estimate"
            if self.tokens.encoder
            else "offline-byte-estimate",
            "duplicates_removed": len(duplicates),
            "duplicate_ids": duplicates,
            "selected_ids": selected_ids,
            "omitted_ids": omitted_ids,
            "recovery_keys": recovery,
            "quality_gates": {
                "required_items_preserved": all(
                    not item["required"] or item["id"] in selected_ids for item in unique
                ),
                "budget_respected_or_mandatory_overflow": (
                    after <= token_budget or mandatory_overflow
                ),
                "recoverable_omissions": all(
                    item_id in recovery for item_id in omitted_ids + duplicates
                ),
            },
        }
        return ContextPlanResult(context=context, receipt=receipt)


class LocalContextCache:
    """Content-addressed cache. The content itself is not sent to any service."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def remember(self, content: str) -> CacheResult:
        raw = str(content).encode("utf-8")
        key = hashlib.sha256(raw).hexdigest()
        data = self._load()
        hit = key in data
        tokens = TokenEstimator().count(content)
        data[key] = {"tokens": tokens, "last_seen": time.time()}
        _atomic_write(self.path, data)
        return CacheResult(key=key, hit=hit, tokens=tokens)


class TaskStore:
    """Small local task ledger; intentionally offline and dependency-free."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _load(self) -> list[Task]:
        if not self.path.exists():
            return []
        try:
            rows = json.loads(self.path.read_text(encoding="utf-8"))
            return [Task(**row) for row in rows]
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return []

    def _save(self, tasks: list[Task]) -> None:
        _atomic_write(self.path, [asdict(task) for task in tasks])

    def add(self, title: str, priority: str = "normal") -> Task:
        if not str(title).strip():
            raise ValueError("task title cannot be empty")
        tasks = self._load()
        now = time.time()
        task = Task(
            id=hashlib.sha1(f"{title}:{now}".encode()).hexdigest()[:10],
            title=str(title).strip(),
            priority=priority,
            created_at=now,
            updated_at=now,
        )
        tasks.append(task)
        self._save(tasks)
        return task

    def update(self, task_id: str, **changes: str) -> Task:
        tasks = self._load()
        allowed = {"title", "status", "priority"}
        for task in tasks:
            if task.id == task_id:
                for key, value in changes.items():
                    if key in allowed:
                        setattr(task, key, value)
                task.updated_at = time.time()
                self._save(tasks)
                return task
        raise KeyError(f"task not found: {task_id}")

    def list(self) -> list[Task]:
        return self._load()


class SavingsLedger:
    """Append-only local receipts for reproducible savings reporting."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def record(self, receipt: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "recorded_at": time.time(),
            "input_tokens_before": int(receipt.get("input_tokens_before", 0)),
            "input_tokens_after": int(receipt.get("input_tokens_after", 0)),
            "transformations": list(receipt.get("transformations", [])),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def summary(self) -> dict[str, int | float]:
        requests = before = after = 0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    before += int(row.get("input_tokens_before", 0))
                    after += int(row.get("input_tokens_after", 0))
                    requests += 1
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
        saved = max(0, before - after)
        return {
            "requests": requests,
            "tokens_before": before,
            "tokens_after": after,
            "tokens_saved": saved,
            "reduction_pct": round(saved / before * 100, 2) if before else 0.0,
        }


def run_benchmark(store_path: Path) -> dict[str, Any]:
    """Run a deterministic offline benchmark with explicit quality gates."""
    store = ContentStore(Path(store_path))
    optimizer = FreeshaOptimizer()

    json_value = {
        "task": "classify-events",
        "events": [
            {
                "id": index,
                "kind": "worker_event",
                "status": "processed",
                "retryable": False,
            }
            for index in range(80)
        ],
    }
    json_source = json.dumps(json_value, ensure_ascii=False, indent=4)
    json_result = optimizer.optimize_json(json_source)

    log_lines = [
        (
            f"2026-07-18T12:00:{index % 60:02d}Z INFO worker "
            f"job={5000 + index} completed in {index + 1}ms"
        )
        for index in range(200)
    ]
    critical_line = "2026-07-18T12:03:20Z ERROR request_id=req-benchmark database timeout"
    log_lines.insert(137, critical_line)
    log_source = "\n".join(log_lines)
    log_result = ToolOutputCompactor(store).compact(log_source)

    context_items: list[dict[str, Any]] = [
        {
            "id": "contract",
            "content": "Keep the public API stable and preserve request identifiers.",
            "required": True,
            "priority": "critical",
        },
        {
            "id": "incident",
            "content": "ERROR req-benchmark database timeout in payment worker",
            "priority": "high",
        },
    ]
    context_items.extend(
        {
            "id": f"duplicate-{index}",
            "content": "Routine worker heartbeat completed successfully.",
        }
        for index in range(60)
    )
    context_items.extend(
        {
            "id": f"noise-{index}",
            "content": (
                f"Unrelated design review item {index} about spacing, color, "
                "and optional presentation details."
            ),
            "priority": "low",
        }
        for index in range(20)
    )
    context_result = EconomyContextPlanner(store).build(
        {
            "task": "Investigate req-benchmark payment database timeout",
            "items": context_items,
        },
        token_budget=220,
    )

    before = (
        json_result.original_tokens
        + log_result.original_tokens
        + int(context_result.receipt["input_tokens_before"])
    )
    after = (
        json_result.optimized_tokens
        + log_result.compacted_tokens
        + int(context_result.receipt["input_tokens_after"])
    )
    gates = {
        "json_equivalent": json.loads(json_result.optimized) == json_value,
        "log_exactly_recoverable": bool(
            log_result.recovery_key and store.get(log_result.recovery_key) == log_source
        ),
        "critical_error_preserved": critical_line in log_result.output,
        "required_context_preserved": "contract" in context_result.receipt["selected_ids"],
        "relevant_context_preserved": "incident" in context_result.receipt["selected_ids"],
        "context_budget_respected": bool(
            context_result.receipt["quality_gates"]["budget_respected_or_mandatory_overflow"]
        ),
        "omissions_recoverable": bool(
            context_result.receipt["quality_gates"]["recoverable_omissions"]
        ),
        "net_loss_gate": not ToolOutputCompactor(store).compact("ok").changed,
    }
    gates["all_passed"] = all(gates.values())
    saved = max(0, before - after)
    return {
        "benchmark": "freesha-economy-v1",
        "network_calls": 0,
        "tokenizer": context_result.receipt["tokenizer"],
        "aggregate": {
            "tokens_before": before,
            "tokens_after": after,
            "tokens_saved": saved,
            "reduction_pct": round(saved / before * 100, 2) if before else 0.0,
        },
        "scenarios": {
            "lossless_json": {
                "tokens_before": json_result.original_tokens,
                "tokens_after": json_result.optimized_tokens,
                "reduction_pct": json_result.reduction_pct,
            },
            "recoverable_logs": {
                "tokens_before": log_result.original_tokens,
                "tokens_after": log_result.compacted_tokens,
                "reduction_pct": log_result.reduction_pct,
            },
            "budgeted_context": {
                "tokens_before": context_result.receipt["input_tokens_before"],
                "tokens_after": context_result.receipt["input_tokens_after"],
                "reduction_pct": context_result.receipt["reduction_pct"],
                "duplicates_removed": context_result.receipt["duplicates_removed"],
            },
        },
        "quality_gates": gates,
        "claim_scope": (
            "Measured on the bundled deterministic fixtures; workload-specific, "
            "not a guarantee for every request."
        ),
    }


def forward_chat(
    payload: dict[str, Any],
    api_key: str | None = None,
    endpoint: str | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """Explicit opt-in OpenAI-compatible forwarding; no network call by optimization alone."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required only when --forward is used")
    url = endpoint or os.getenv("FREESHA_ENDPOINT", "https://api.openai.com/v1/chat/completions")
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"upstream HTTP {exc.code}: {body[:500]}") from exc


def _demo() -> None:
    optimizer = FreeshaOptimizer()
    payload = {
        "model": "gpt-5.6",
        "messages": [
            {
                "role": "user",
                "content": json.dumps(
                    {"task": "classify", "items": [{"id": i, "text": "signal"} for i in range(10)]},
                    indent=4,
                ),
            }
        ],
    }
    result = optimizer.optimize_request(payload)
    print(
        json.dumps(
            {"optimized_payload": result.payload, "receipt": result.receipt},
            ensure_ascii=False,
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freesha local-first context optimizer")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo")
    optimize = sub.add_parser("optimize-json")
    optimize.add_argument("path", nargs="?", help="JSON file; stdin when omitted")
    skeleton = sub.add_parser("skeleton")
    skeleton.add_argument("path")
    compact = sub.add_parser(
        "compact-output", help="compact repetitive tool/log output recoverably"
    )
    compact.add_argument("path", nargs="?", help="text file; stdin when omitted")
    compact.add_argument("--store", default=".freesha/blobs")
    recover = sub.add_parser("recover", help="restore exact locally stored content")
    recover.add_argument("key")
    recover.add_argument("--store", default=".freesha/blobs")
    economy = sub.add_parser("economy", help="build a deduplicated, budgeted context packet")
    economy.add_argument("path", help="JSON packet with task and items")
    economy.add_argument("--budget", type=int, default=2000)
    economy.add_argument("--store", default=".freesha/blobs")
    cache = sub.add_parser(
        "prepare-openai-cache", help="add documented GPT-5.6 cache routing fields"
    )
    cache.add_argument("path", help="OpenAI request JSON")
    cache.add_argument("--key", required=True)
    benchmark = sub.add_parser("benchmark", help="run the offline economy benchmark")
    benchmark.add_argument("--store", default=".freesha/benchmark-blobs")
    joy = sub.add_parser(
        "joy", help="preview experimental fixture-based model routing (dry-run only)"
    )
    joy.add_argument("path", help="JOY task JSON")
    joy.add_argument("--models", required=True, help="fixture-only model catalog JSON")
    joy.add_argument("--dry-run", action="store_true")
    joy.add_argument("--store", default=".freesha/joy-blobs")
    task = sub.add_parser("task")
    task_sub = task.add_subparsers(dest="task_command", required=True)
    task_add = task_sub.add_parser("add")
    task_add.add_argument("title")
    task_add.add_argument("--priority", default="normal")
    task_list = task_sub.add_parser("list")
    task_list.add_argument("--path", default=".freesha/tasks.json")
    task_update = task_sub.add_parser("update")
    task_update.add_argument("id")
    task_update.add_argument("--path", default=".freesha/tasks.json")
    task_update.add_argument("--status")
    task_update.add_argument("--priority")
    ledger = sub.add_parser("ledger")
    ledger.add_argument("--path", default=".freesha/receipts.jsonl")
    args = parser.parse_args(argv)

    optimizer = FreeshaOptimizer()
    if args.command == "demo":
        _demo()
    elif args.command == "optimize-json":
        source = Path(args.path).read_text(encoding="utf-8") if args.path else sys.stdin.read()
        result = optimizer.optimize_json(source)
        print(
            json.dumps(
                {"optimized": result.optimized, "receipt": asdict(result)},
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "skeleton":
        print(optimizer.python_skeleton(Path(args.path).read_text(encoding="utf-8")))
    elif args.command == "compact-output":
        source = Path(args.path).read_text(encoding="utf-8") if args.path else sys.stdin.read()
        result = ToolOutputCompactor(ContentStore(Path(args.store))).compact(source)
        print(
            json.dumps(
                {
                    "output": result.output,
                    "receipt": {
                        "mode": result.mode,
                        "changed": result.changed,
                        "original_tokens": result.original_tokens,
                        "compacted_tokens": result.compacted_tokens,
                        "tokens_saved": result.tokens_saved,
                        "reduction_pct": result.reduction_pct,
                        "recovery_key": result.recovery_key,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "recover":
        sys.stdout.write(ContentStore(Path(args.store)).get(args.key))
    elif args.command == "economy":
        packet = json.loads(Path(args.path).read_text(encoding="utf-8"))
        result = EconomyContextPlanner(ContentStore(Path(args.store))).build(
            packet, token_budget=args.budget
        )
        print(
            json.dumps(
                {"context": result.context, "receipt": result.receipt},
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "prepare-openai-cache":
        payload = json.loads(Path(args.path).read_text(encoding="utf-8"))
        result = optimizer.prepare_openai_cache(payload, cache_key=args.key)
        print(
            json.dumps(
                {"payload": result.payload, "receipt": result.receipt},
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "benchmark":
        print(json.dumps(run_benchmark(Path(args.store)), ensure_ascii=False, indent=2))
    elif args.command == "joy":
        if not args.dry_run:
            parser.error("experimental JOY requires --dry-run; live execution is unavailable")
        from freesha_joy import JoyRouter, load_json_object

        try:
            task_payload = load_json_object(Path(args.path))
            model_catalog = load_json_object(Path(args.models))
        except OSError:
            parser.error("JOY could not read one of the input files")
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            parser.error(f"JOY input validation failed: {exc}")
        try:
            receipt = JoyRouter(ContentStore(Path(args.store))).route(
                task_payload, model_catalog
            )
        except (TypeError, ValueError) as exc:
            parser.error(f"JOY input validation failed: {exc}")
        print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
        if not receipt["route_found"]:
            return 2
    elif args.command == "task":
        task_path = Path(getattr(args, "path", ".freesha/tasks.json"))
        store = TaskStore(task_path)
        if args.task_command == "add":
            print(
                json.dumps(
                    asdict(store.add(args.title, args.priority)), ensure_ascii=False, indent=2
                )
            )
        elif args.task_command == "list":
            print(json.dumps([asdict(task) for task in store.list()], ensure_ascii=False, indent=2))
        elif args.task_command == "update":
            changes = {
                key: value
                for key, value in {"status": args.status, "priority": args.priority}.items()
                if value is not None
            }
            print(
                json.dumps(asdict(store.update(args.id, **changes)), ensure_ascii=False, indent=2)
            )
    elif args.command == "ledger":
        print(json.dumps(SavingsLedger(Path(args.path)).summary(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
