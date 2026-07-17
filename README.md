# Freesha

**Project name:** Freesha

**Elevator pitch (under 200 characters):**

> Freesha is a local-first context layer for Codex and OpenAI-compatible apps that structures tasks, removes lossless payload waste, reuses repeated context, and proves token savings before sending.

## The problem

AI agents do not only spend tokens on the answer. They repeatedly resend verbose JSON, unchanged context, tool output, and task history. Most “token saver” demos either change the model's behavior with a prompt or make lossy compression claims without a receipt.

Freesha takes a narrower approach: **optimize what can be optimized deterministically, keep an audit receipt, and never make a network request unless the caller explicitly asks for forwarding.**

## What works today

- **Lossless JSON minification** inside message content. Parsed JSON is serialized without whitespace and remains semantically identical.
- **Token receipt** with before/after estimates, transformations, and tokenizer provenance.
- **Content-addressed local context cache**. Repeated identical context is recognized locally without re-sending it to Freesha or a provider.
- **Python structure view**. Extracts classes, callable signatures, and control-flow headers instead of sending an entire file when a map is enough.
- **Local task ledger**. Creates and updates small structured tasks without an LLM call.
- **Explicit OpenAI-compatible forwarding helper**. `forward_chat()` only sends when the caller supplies `OPENAI_API_KEY`; optimization itself is offline.
- **Optional compact-output prompt presets** (`lite`, `full`, `ultra`) that preserve code, commands, and errors.

## Quick start

Requires Python 3.11+. There are no mandatory dependencies.

```bash
git clone https://github.com/rexamax/freesha.git
cd freesha
python3 -m unittest discover -s tests -v
python3 freesha_core.py demo
```

For more accurate OpenAI-family token counts, install the optional tokenizer:

```bash
python3 -m pip install -e '.[tokens]'
```

Optimize a JSON file without calling an API:

```bash
python3 freesha_core.py optimize-json payload.json
```

Print a Python structure map:

```bash
python3 freesha_core.py skeleton src/example.py
```

Create and update local structured tasks without an LLM request:

```bash
python3 freesha_core.py task add "Prepare Build Week demo" --priority high
python3 freesha_core.py task list
python3 freesha_core.py ledger
```

Use the library:

```python
from freesha_core import FreeshaOptimizer

payload = {
    "model": "gpt-5.6",
    "messages": [{
        "role": "user",
        "content": '{"events": [  { "id": 1, "kind": "signal" }  ]}',
    }],
}

result = FreeshaOptimizer().optimize_payload(payload)
print(result.receipt)
# Forward only after an explicit opt-in:
# from freesha_core import forward_chat
# response = forward_chat(result.payload)
```

## Architecture

```text
caller / Codex helper / app
          |
          v
  FreeshaOptimizer
  ├── content detector
  ├── lossless JSON minifier
  ├── Python structure extractor (explicit mode)
  ├── local SHA-256 context cache
  ├── token receipt / savings ledger
  └── optional OpenAI-compatible forwarder
          |
          v
  OpenAI / Codex-compatible endpoint (opt-in only)
```

Freesha does **not** pretend that arbitrary prose can be compressed losslessly by deleting words. Free-text compression is a future, opt-in A/B experiment and must be quality-gated. Current automatic optimization is intentionally conservative.

## What we learned from the reference projects

- **Caveman:** output-style control can reduce ceremony, but a system prompt is not input compression and does not prove savings. Freesha keeps the idea as explicit output presets and records no fake percentage.
- **Headroom:** the important architectural idea is a local content router plus reversible/cache-aware context handling. Freesha starts with a smaller dependency-free core instead of copying its model-based compressors.
- **LeanCTX:** signatures, selective reads, caching, and receipts are more valuable than blindly minifying every string. Freesha's Python skeleton is explicit because a structure map is not a substitute for source code when implementation details matter.

Freesha is an independent implementation. It does not copy code from those repositories.

## Codex and GPT-5.6 usage

This project is being developed with Codex during OpenAI Build Week. GPT-5.6/Codex is used for the implementation, test design, debugging, and review of the optimization pipeline. The README and demo claims are intentionally limited to behavior that can be exercised locally.

For the Devpost submission, add the actual Codex feedback/session identifier from the primary Codex build session:

```text
/feedback <SESSION_ID_FROM_CODEX>
```

The identifier must be obtained inside the Codex app/session where the majority of the project was built. A Hermes chat ID is not a substitute.

## Measurement plan

Every optimization produces a receipt. The first benchmark target is not “70%” in the abstract; it is:

1. byte/token reduction on representative Telegram/news JSON payloads;
2. exact `json.loads(before) == json.loads(after)` for lossless JSON transforms;
3. unchanged results on a fixed classification fixture;
4. cache hit rate on repeated context;
5. no network request during local optimization.

The current test suite proves the first vertical slice. More real OpenAI A/B runs should be added before making quality or dollar-savings claims.

## Hackathon fit

Primary track: **Work and Productivity**. Secondary fit: **Developer Tools**.

The demo should be under three minutes:

1. Show a verbose JSON payload and run Freesha offline.
2. Show the receipt: before/after estimate and exact JSON equivalence.
3. Repeat the same context and show a local cache hit.
4. Show the Python structure view.
5. Show that forwarding is opt-in and that no key is required for local optimization.
6. Explain where Codex/GPT-5.6 helped build and test the project.

## Security and privacy

- No telemetry or accounts are needed for local features.
- No API key is hardcoded or required for optimization.
- Context cache stores hashes and small metadata, not the original content.
- Do not commit `.env`, provider keys, session files, or private project data.
- Network forwarding is explicit and uses the standard `OPENAI_API_KEY` environment variable.

## Status

This is a working hackathon MVP, not a finished universal proxy. The next high-value additions are a local HTTP compatibility endpoint, reversible retrieval by hash, a fixed A/B quality benchmark, and a small dashboard for receipts.

## License

MIT.
