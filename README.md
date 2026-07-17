# Freesha 🧠📉

An intelligent proxy layer for OpenAI and OpenRouter APIs that slashes token costs by up to 70% through dynamic context parsing and automated payload minification.

## Inspiration
As developers integrating LLMs into daily workflows, we hit a massive bottleneck: context costs. When trying to extract insights from massive datasets—like parsing a million-message private chat archive or continuously monitoring automated news feeds—token costs spiral out of control. We built a solution that acts as a smart "diet" for LLMs.

## Core Features (The Token-Saving Trinity)

1. **Caveman Mode:** Enforces strict system prompts that command the model to output purely structured data without conversational filler or markdown formatting.
2. **Headroom Compression:** Automatically detects the payload type and minifies heavy structures (like massive JSON exports) on the fly, removing whitespaces and redundant keys.
3. **Lean-ctx (Contextual Skeleton):** Instead of sending entire files or data dumps, Freesha extracts and sends only the AST (Abstract Syntax Tree) or structural headers. The model requests the full body only if explicitly needed.

## Tech Stack
`Python`, `OpenAI API`, `AST Parsing`, `JSON Minification`

## How Codex & GPT-5.6 are used
Freesha is designed to act as a middleware for OpenAI's most advanced models. We use **Codex** to automatically test and refine our AST parsing logic (Lean-ctx), ensuring the structural extraction of Python code is perfectly formatted for LLM consumption. **GPT-5.6** is utilized as the primary engine for analyzing the compressed payloads, proving that our token-reduction strategies (Headroom & Caveman mode) do not degrade the model's analytical capabilities even when processing complex, minified JSON architectures.
