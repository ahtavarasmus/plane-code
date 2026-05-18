# plane-code

An ontology-based coding agent for Rust workspaces. Indexes your crate
into a structural graph (functions, types, traits, modules, files) and
drives an LLM through a tight query / edit / verify loop with hard
guardrails against the model editing code it hasn't actually read.

The package on crates.io is `planecode`. The installed binary is
`plane-code`.

## Install

```bash
cargo install planecode
```

This builds from source, so you need a Rust toolchain. The installed
binary is on your PATH as `plane-code`.

## Run it

From any Rust workspace:

```bash
plane-code
```

That brings up an interactive REPL. Type a request, the agent indexes
the workspace, queries the graph, makes edits, and runs cargo to verify.

Single-shot mode:

```bash
plane-code -p "rename Foo to Bar in src/auth"
```

## Models

plane-code talks to two backends:

- **Ollama** (local, default) — run `ollama serve` and the daemon at
  `http://localhost:11434`. Pull a model first, e.g.
  `ollama pull qwen3:32b`. Tested with qwen3:8b, qwen3:32b, qwen3.6:27b.
- **Groq** (hosted, fast) — pass `--provider groq` and set `GROQ_API_KEY`
  (or `--api-key`). Tested with `qwen/qwen3-32b`.

Switch backends inside the REPL with `/model`. The selection persists
across launches in `~/.plane-code/config.json`.

## What's the loop

Three tools, one phase each, in order:

- `query_ontology` — explore the graph. Find functions by keyword,
  filter by path, traverse callers/callees/tests/fields/impls.
- `update_ontology` — edit. `replace_body`, `replace_item`, `rename`,
  `add_function`, `edit_file`, `create_file`, `delete_file`.
- `run_cargo` — verify. Compile errors are classified as
  introduced-by-this-edit vs pre-existing so the model only chases its
  own breakage.

A fourth tool, `show_flow`, opens an interactive control-flow diagram
in your browser when the agent wants to show you how some piece of code
is wired (a click-to-expand view, faster than narrating in prose).

## Read-before-edit guardrail

Small LLMs love to invent APIs. They'll write `Foo::new()` against a
struct that has no `new`, reach for enum variants that don't exist, or
guess a method's signature. The harness blocks this:

Every `update_ontology` call refuses to dispatch if the target entity
(or file) hasn't been seen by a prior `query_ontology` in this session.
The rollback hint points at the exact query call to make first. After
a successful edit, the target is freshened in the read set (the diff
in the response covers the new state), so chained edits don't need
re-reads.

Module-level queries don't count as reads — they only show signatures,
not bodies. Function / Type / Trait / File queries do, because their
responses carry the actual source.

## Slash commands

- `/help` — list commands
- `/model` — switch provider or model
- `/think` — toggle reasoning visibility
- `/trace` — toggle verbose tracing
- `/resume` — pick a prior session to continue
- `/clear` — reset conversation (keeps the system prompt)
- `/reindex` — re-walk the workspace
- `/export` — dump the current session to JSON

## Session persistence

Conversations are saved per workspace under
`~/.plane-code/sessions/<workspace-hash>/<id>.json`. Latest 30 sessions
per workspace are kept; older ones are pruned automatically.

## Development

```bash
git clone https://github.com/ahtavarasmus/plane-code
cd plane-code
cargo build --release
./target/release/plane-code
```

## License

Dual-licensed under MIT or Apache-2.0, at your option. See `LICENSE-MIT`
and `LICENSE-APACHE`.
