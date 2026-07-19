# Freesha future roadmap (draft)

This document separates shipped behavior from experiments and possible future work. It is not a delivery schedule.

## 1. Current MVP

The current stable product is a local CLI and Python API for lossless JSON minification, byte-exact deduplication, recoverable log compaction, deterministic context budgeting, local SHA-256 recovery, net-loss passthrough, receipts, and explicit prompt-cache request preparation. Optimization and the bundled benchmark require no API key and make no network request.

The public [MVP demo](https://youtu.be/Vn4Kng-07W4) remains accurate for this stable boundary.

## 2. JOY dry-run experiment

JOY (Justify, Optimize, Yield) is implemented only as a local deterministic preview. It consumes a user-supplied catalog that must declare itself fixture-only, reuses the existing context planner, applies hard quality/capability/privacy/latency/budget filters, and explains the least-cost eligible route. It never checks live availability, sends a provider request, or spends money. Its costs, quality scores, and latency tiers are fixture estimates.

## 3. Shadow evaluation

The next experiment should run JOY recommendations in shadow against a versioned real-task suite while the production model choice remains unchanged. Baseline and candidate must use the same task, model settings, correctness checks, and provider-authoritative token accounting. A recommendation is useful only when quality stays above the floor and total cost—including retries and escalation—falls.

## 4. Opt-in live routing, later

Live routing should remain unavailable until provider usage receipts, model availability configuration, fixed quality gates, privacy enforcement, safe invalidation, and reviewed escalation rules exist. Any later live mode must require explicit network and spend opt-in, environment-supplied credentials, bounded budgets, and a reviewable decision receipt. It must fail closed.

## 5. Local UI and receipt viewer

Based on early peer feedback, a planned local UI will make Freesha's before/after context, recovery receipts, quality gates, and JOY route explanations accessible to users who do not want to work through the CLI.

The UI is not implemented, designed, validated, customer-requested, or scheduled. It should not precede the shadow-evaluation and receipt work above.
