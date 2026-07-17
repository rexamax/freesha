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
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import tiktoken  # Optional: exact tokenizer for cl100k-compatible models.
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
                self.encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.encoder = None

    def count(self, value: Any) -> int:
        text = value if isinstance(value, str) else json.dumps(
            value, ensure_ascii=False, separators=(",", ":")
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
        optimized = json.dumps(
            parsed, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
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
            "estimated_input_reduction_pct": round(
                max(0, before - after) / before * 100, 2
            ) if before else 0.0,
            "tokenizer": "tiktoken/cl100k_base" if self.tokens.encoder else "offline-byte-estimate",
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
                elif isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)):
                    # Show control-flow headers but do not recursively dump implementation.
                    segment = ast.get_source_segment(source, node) or type(node).__name__
                    header = segment.splitlines()[0].strip()
                    lines.append(pad + header + " …")

        visit(tree.body)
        return "\n".join(lines)

    def compact_system_prompt(self, level: str = "full") -> str:
        levels = {
            "lite": "Be concise. Skip pleasantries. Preserve code, commands, errors.",
            "full": "Answer directly. No pleasantries or repeated context. Preserve code, commands, errors exactly.",
            "ultra": "Direct answer only. Preserve code, commands, errors exactly.",
        }
        if level not in levels:
            raise ValueError("level must be lite, full, or ultra")
        return levels[level]

    def optimize_request(self, payload: dict[str, Any], output_mode: str = "full") -> PayloadResult:
        result = self.optimize_payload(payload)
        if output_mode:
            result.payload.setdefault("metadata", {})
            result.payload["metadata"]["freesha_output_mode"] = output_mode
        return result


def _atomic_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


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
            title=str(title).strip(), priority=priority, created_at=now, updated_at=now,
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


def forward_chat(payload: dict[str, Any], api_key: str | None = None, endpoint: str | None = None, timeout: int = 60) -> dict[str, Any]:
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
        "messages": [{
            "role": "user",
            "content": json.dumps({"task": "classify", "items": [{"id": i, "text": "signal"} for i in range(10)]}, indent=4),
        }],
    }
    result = optimizer.optimize_request(payload)
    print(json.dumps({"optimized_payload": result.payload, "receipt": result.receipt}, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freesha local-first context optimizer")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo")
    optimize = sub.add_parser("optimize-json")
    optimize.add_argument("path", nargs="?", help="JSON file; stdin when omitted")
    skeleton = sub.add_parser("skeleton")
    skeleton.add_argument("path")
    args = parser.parse_args(argv)

    optimizer = FreeshaOptimizer()
    if args.command == "demo":
        _demo()
    elif args.command == "optimize-json":
        source = Path(args.path).read_text(encoding="utf-8") if args.path else sys.stdin.read()
        result = optimizer.optimize_json(source)
        print(json.dumps({"optimized": result.optimized, "receipt": asdict(result)}, ensure_ascii=False, indent=2))
    elif args.command == "skeleton":
        print(optimizer.python_skeleton(Path(args.path).read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
