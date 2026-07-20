# Freesha

**Local-first economy mode for OpenAI-compatible AI workflows.**

Freesha reduces avoidable input tokens before an API call, keeps large omitted data recoverable on the local machine, and emits receipts instead of unsupported savings claims.

> Bundled offline benchmark: **76.68% aggregate reduction** with the dependency-free fallback estimator and **79.59%** with optional `tiktoken/o200k_base`, with all deterministic quality gates passing. These are workload-specific local estimates—not a guarantee for every request and not a provider billing receipt.

Public MVP demo: [YouTube](https://youtu.be/Vn4Kng-07W4). The video demonstrates the stable economy pipeline; the JOY routing preview described below is a separate experiment added afterward.

## 60-second judge check (no API key)

Linux/macOS:

```bash
python3 -m unittest discover -s tests
python3 freesha_core.py benchmark --store .freesha/judge-benchmark
python3 freesha_core.py joy examples/joy_task.json --models examples/joy_models.fixture.json --dry-run --store .freesha/judge-joy
```

Windows PowerShell or CMD:

```powershell
py -3.11 -m unittest discover -s tests
py -3.11 freesha_core.py benchmark --store .freesha/judge-benchmark
py -3.11 freesha_core.py joy examples/joy_task.json --models examples/joy_models.fixture.json --dry-run --store .freesha/judge-joy
```

All three commands are local and deterministic. Check `quality_gates.all_passed: true` in the benchmark, then `route_found: true`, `network_calls: 0`, `provider_request_sent: false`, and `automatic_spend: false` in the JOY receipt.

## Product boundary

| Status | Capability |
| --- | --- |
| Current stable MVP | Lossless JSON minification, recoverable log compaction, exact deduplication, budgeted context selection, local recovery, receipts, net-loss passthrough, and opt-in prompt-cache request preparation. |
| Experimental now | JOY fixture-based routing preview: local, deterministic, dry-run only, and unable to execute provider calls or spend. |
| Roadmap, not implemented | Real-task shadow evaluation, provider-authoritative usage receipts, opt-in live routing, transparent gateway integration, and a local receipt UI. |

## Why it exists

AI apps repeatedly send waste that does not improve the answer:

- pretty-printed JSON;
- repeated status and tool-output lines;
- duplicate context items;
- irrelevant context that does not fit the current task;
- unstable prompt layouts that prevent provider cache hits.

Freesha applies the cheapest safe operation first and passes through input when an optimization would be a net loss.

## Economy pipeline

```text
input
  ├─ lossless JSON minification
  ├─ byte-exact context deduplication
  ├─ recoverable log-template compaction
  ├─ required + task-relevant context selection
  ├─ token budget / net-loss gates
  └─ optional documented GPT-5.6 prompt-cache fields
       ↓
optimized context + local recovery blobs + receipt
       ↓
API call only when the host application explicitly makes one
```

### Safety properties

- **Local by default:** optimization and benchmark commands make zero network calls.
- **Recoverable and integrity-checked:** compressed or omitted content is stored under `.freesha/blobs/` by SHA-256, verified against its recovery key on read, and restored byte-for-byte. A modified blob fails closed instead of being returned as valid.
- **Atomic/concurrency-safe state writes:** cache and task updates use OS-level file locks, unique temporary files, file `fsync`, atomic replacement, and parent-directory `fsync` where the platform supports it. Syntactically or semantically malformed state fails closed and is left untouched.
- **Critical-line preservation:** errors, failures, warnings, exceptions, panics, access denials, and timeouts are kept verbatim during log compaction.
- **Net-loss gate:** short or incompressible output is returned unchanged.
- **No unknown request fields:** the old custom `metadata` injection was removed.
- **No telemetry, accounts, or mandatory dependencies.**

## Quick start

Python 3.11+ is required. The default path uses only the Python standard library. The CLI is intended for Linux, macOS, and Windows and is checked by a three-OS GitHub Actions matrix.

Install from a clean checkout:

```bash
git clone https://github.com/rexamax/freesha.git
cd freesha
python3 -m unittest discover -s tests -v
python3 freesha_core.py benchmark
```

Optional local tokenizer approximation:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[tokens]'
freesha benchmark
freesha joy --dry-run
```

`tiktoken/o200k_base` is labeled as a **local estimate**. OpenAI's token-count endpoint and response `usage` fields are authoritative for provider billing.

## Current benchmark

Run:

```bash
python3 freesha_core.py benchmark
```

Expected shape (exact token counts can differ depending on whether optional `tiktoken` is installed):

```json
{
  "benchmark": "freesha-economy-v1",
  "network_calls": 0,
  "aggregate": {
    "reduction_pct": 79.59
  },
  "scenarios": {
    "lossless_json": {"reduction_pct": 45.41},
    "recoverable_logs": {"reduction_pct": 97.13},
    "budgeted_context": {"reduction_pct": 82.02}
  },
  "quality_gates": {
    "json_equivalent": true,
    "log_exactly_recoverable": true,
    "critical_error_preserved": true,
    "required_context_preserved": true,
    "relevant_context_preserved": true,
    "context_budget_respected": true,
    "omissions_recoverable": true,
    "net_loss_gate": true,
    "all_passed": true
  }
}
```

The benchmark intentionally reports each scenario separately. JSON whitespace removal may save much less than repetitive logs; a short prompt may save nothing.

## Experimental JOY routing preview

JOY means **Justify, Optimize, Yield**. JOY is experimental, local, deterministic, and does not execute provider calls. It first asks the existing economy planner for an optimized local context estimate, rejects every model fixture that violates a hard task constraint, then ranks the remaining entries by projected input cost, latency tier, and model ID.

```bash
freesha joy --dry-run
```

The installed command uses bundled synthetic fixtures. From a checkout, explicit fixture paths remain supported:

```bash
python3 freesha_core.py joy examples/joy_task.json \
  --models examples/joy_models.fixture.json --dry-run
```

Representative excerpt from the dependency-free run of the committed synthetic fixture (the optional tokenizer can change the estimate):

```json
{
  "mode": "joy-dry-run",
  "experimental": true,
  "selected_model": {
    "id": "fixture/economy",
    "estimated_input_tokens": 127,
    "projected_input_cost": 0.0000254,
    "currency": "USD"
  },
  "network_calls": 0,
  "provider_request_sent": false,
  "automatic_spend": false
}
```

The catalog is explicitly fixture-only. Its model IDs, quality scores, latency tiers, and costs are synthetic estimates—not current provider rankings, availability, pricing, or a recommendation to spend. Omitting `--dry-run` fails closed. A successful preview returns a safe task fingerprint, constraint decisions, rejection reasons, higher-quality fallback candidates, and local recovery handles without serializing the raw task or context.

## CLI

### 1. Compact tool or log output

```bash
python3 freesha_core.py compact-output examples/repetitive.log
```

The JSON result contains compacted output, before/after token estimates, reduction percentage, and a `recovery_key`.

Restore the original exactly:

```bash
python3 freesha_core.py recover <RECOVERY_KEY>
```

### 2. Build an economy context packet

```bash
python3 freesha_core.py economy examples/context_packet.json --budget 48
```

Input contract:

```json
{
  "task": "Investigate checkout timeout request req-42",
  "items": [
    {
      "id": "contract",
      "content": "Keep the public API stable.",
      "required": true,
      "priority": "critical"
    },
    {
      "id": "incident",
      "content": "ERROR req-42 checkout timeout",
      "kind": "text",
      "priority": "high"
    }
  ]
}
```

Rules:

- `id` must be a unique, short public-safe identifier, not a filesystem path;
- `required` must be a JSON boolean; `required: true` always wins;
- relevance is deterministic lexical overlap plus explicit priority and critical signals;
- `kind: "json"` enables lossless minification;
- `kind: "log"` enables recoverable template compaction;
- only byte-identical optional content is deduplicated; whitespace and formatting differences are preserved;
- duplicate and omitted items receive local recovery keys in the receipt.

### 3. Prepare GPT-5.6 prompt caching

Keep static instructions and tools first, variable user data last. Then run:

```bash
python3 freesha_core.py prepare-openai-cache \
  examples/openai_request.json \
  --key support:v1
```

Freesha adds only documented GPT-5.6 fields:

```json
{
  "prompt_cache_key": "support:v1",
  "prompt_cache_options": {
    "mode": "implicit",
    "ttl": "30m"
  }
}
```

Important: GPT-5.6 cache writes are billed at 1.25× uncached input, while reads can save cost. A stable key is useful only when the same long prefix is reused. Measure `cached_tokens` and `cache_write_tokens`; do not assume caching always saves money.

### 4. Existing utilities

```bash
python3 freesha_core.py optimize-json payload.json
python3 freesha_core.py skeleton src/example.py
python3 freesha_core.py task add "Prepare demo" --priority high
python3 freesha_core.py task list
python3 freesha_core.py ledger
```

## Python API

```python
from pathlib import Path

from freesha_core import ContentStore, EconomyContextPlanner

packet = {
    "task": "Investigate request req-42 timeout",
    "items": [
        {"id": "contract", "content": "Keep API stable.", "required": True},
        {"id": "incident", "content": "ERROR req-42 timeout"},
    ],
}

planner = EconomyContextPlanner(ContentStore(Path(".freesha/blobs")))
result = planner.build(packet, token_budget=500)
print(result.context)
print(result.receipt)
```

## How to prove real API value

A local estimate is not enough for a production claim. Use this A/B protocol:

1. Build a fixed task set with expected facts, IDs, and pass/fail criteria.
2. Send the baseline request and the Freesha request to the **same model and settings**.
3. For exact OpenAI input counts, use `POST /v1/responses/input_tokens` on both request shapes.
4. For live calls, record `input_tokens`, `output_tokens`, `cached_tokens`, `cache_write_tokens`, latency, retries, and price.
5. Run the same quality checks on both answers.
6. Count savings only when the optimized answer still passes the quality gate.
7. Repeat long stable prefixes within the provider cache TTL to measure read/write economics.

The useful metric is:

```text
net value = provider cost avoided - cache-write/compression cost
```

Not:

```text
local characters removed = money saved
```

## What the 70%+ result means

The included mixed fixture currently clears 70% aggregate reduction because it contains the waste Freesha targets: verbose JSON, repetitive tool logs, duplicates, and off-task context. It does **not** mean:

- every request becomes 70% smaller;
- response quality is universally unchanged;
- 70% fewer TPM rate-limit tokens (prompt caching still counts toward TPM);
- 70% lower total bill when output/reasoning dominates;
- local estimates equal provider billing.

## Privacy and storage

- Recovery blobs remain local and are ignored by Git through `.freesha/`.
- Blob filenames are SHA-256 hashes. Freesha requests `0700`/`0600` permissions on POSIX; on Windows it relies on inherited filesystem ACLs, so keep `.freesha/` inside a protected user profile. The store is not encrypted.
- The cache stores no API key.
- `forward_chat()` is explicit opt-in and requires `OPENAI_API_KEY` only at call time. Remote endpoints must use HTTPS; plain HTTP is accepted only for literal loopback hosts used by local compatible servers.
- Do not commit recovery blobs, `.env`, API keys, session files, customer data, or private source material.
- Treat recovery keys as sensitive references when the underlying content is sensitive.

## Public design references

The implementation is independent and does not copy third-party code. Its provider-aware behavior follows current public documentation:

- [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
- [OpenAI token counting](https://developers.openai.com/api/docs/guides/token-counting)
- [OpenAI compaction](https://developers.openai.com/api/docs/guides/compaction)
- [OpenAI cost optimization](https://developers.openai.com/api/docs/guides/cost-optimization)
- [OpenAI tool search](https://developers.openai.com/api/docs/guides/tools-tool-search)

## How Codex and GPT-5.6 were used

Codex accelerated repository inspection, the Windows concurrency audit, vertical RED→GREEN tests, CLI implementation, deterministic benchmark execution, and an independent review of this bounded JOY experiment. GPT-5.6 guidance informed the existing prompt-cache preparation path and the conservative API-contract decisions: supported fields only, stable-prefix planning, provider usage as authority, and no assumption that cached or locally removed tokens equal billed savings.

The product choices remained human decisions: local-first execution, reversible omissions, verbatim critical lines, net-loss passthrough, no hidden calls, no automatic spend, and claims limited to reproducible evidence. The real Codex `/feedback` Session ID belongs in the Devpost submission; it must be generated in the project session and is intentionally neither invented nor stored in this repository.

## Limitations and next high-ROI step

Freesha is a working local core and CLI, not a transparent gateway or native Codex hook. It has no UI, live adaptive routing, or provider-authoritative proof that the local estimates reduce a real bill. JOY does not know current model availability or pricing and cannot execute its suggested route.

The next production milestone should be shadow evaluation against a fixed real-task suite, without provider routing changes. That evaluation should:

1. call the official token-count endpoint only through explicit opt-in;
2. record provider usage receipts separately from local estimates;
3. compare the same model/settings before and after optimization;
4. run fixed correctness and context-recall gates;
5. keep every route recommendation advisory until the evidence is reviewable.

See [the future roadmap](docs/FUTURE_ROADMAP_DRAFT.md) for the staged boundary.

## License

MIT.
