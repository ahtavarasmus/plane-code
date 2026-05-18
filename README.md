# plane-code

A Rust coding agent designed to be useful with local models. The name
is literal: planecode is built so smaller, cheaper, even fully offline
LLMs are reliable enough for real coding work. You can run it on a
plane with the wifi off.

Most agents fail at this because they delegate too much to the model:
pick the right file with grep, guess the right function name, summarise
what you saw. Each delegation is a chance to confabulate, and small
models confabulate constantly. planecode moves those decisions out of
the model and into the harness, so the model gets exact code to operate
on instead of a search problem. Smaller models become tractable.
Coding gets cheaper.

The package on crates.io is `planecode`. The installed binary is
`plane-code`.

## The core idea: AST graph as the only context

planecode indexes your workspace into a structural graph: functions,
types, traits, modules, files, calls, fields, impls, tests. The model
has no grep, no `Read`, no shell. Every lookup goes through
`query_codebase`, which serves results from the graph.

This matters because:

- The model never has to guess where a function lives or what it's
  called. It asks the graph by intent ("verify JWT tokens") and gets
  the actual symbol back with its source, callers, and callees.
- Responses contain only the code that's structurally relevant. A
  query for one function returns that function's body plus its
  neighbourhood as shallow references. The other 800 lines of the file
  stay out of context.
- File-level reads are possible (`File` object type, used for editing
  imports, attributes, top-of-file metadata) but the response
  prestrips every region the AST already indexed. The body of an
  indexed function comes back as a stub pointing at the structural
  query. The model can't accidentally pull a whole file in when the
  question was about one method.
- Edits are addressed by ontology identifier (name + module_path),
  not file path + line. A rename rewrites every reference atomically.

Each of these shifts a decision the model used to own onto the
harness, which is deterministic and reads the AST. The model just
expresses intent.

## Read-before-edit, enforced

Small models love to invent APIs: call `Foo::new()` on a struct that
has no `new`, reach for enum variants that don't exist, guess a
method signature. The harness rejects every `update_codebase` call
whose target wasn't read by a prior `query_codebase` in the session.
The rollback hint names the exact query to run first. After a
successful edit the diff freshens the read set so chained edits
don't need re-reads.

## Visual verification: show, don't narrate

When the agent wants to explain how code is wired, it calls
`show_flow` instead of describing it in prose. That opens an
interactive control-flow diagram in your browser. Clickable nodes,
expand on click, deterministically derived from the AST so it can't
lie about what's there. The model picks what to show; the AST
decides what the picture looks like.

You can invoke the same view yourself: `/flow <fn>` for one function,
`/skyline` for a whole-workspace map. Use it to verify the model's
claims at a glance instead of re-reading source.

## Install

```bash
cargo install planecode
```

Needs a Rust toolchain. The installed binary lands on your PATH as
`plane-code`.

## Run

From any Rust workspace:

```bash
plane-code
```

Interactive REPL. Or single-shot:

```bash
plane-code -p "rename Foo to Bar in src/auth"
```

## Models

Two backends, both selectable at runtime via `/model`. The picker
fetches model lists live from each provider, so no hardcoded list
goes stale.

- **Ollama** (local, default). Run `ollama serve`. The picker shows
  every downloaded model alongside the public registry; selecting an
  un-downloaded one drills into a tag-sized picker and pulls the
  variant you choose. Tested seriously with qwen3:8b on a laptop and
  qwen3:32b on a desktop GPU.
- **Groq** (hosted, fast). Set `GROQ_API_KEY` or pass `--api-key`,
  then pick from the live catalogue.

The selection persists at `~/.plane-code/config.json` across launches.

## The agent's tools

Four tools, one phase each:

- `query_codebase` - traverse the graph. Find by keyword, filter by
  path, expand callers / callees / tests / fields / impls /
  implementors.
- `update_codebase` - structural edits: `replace_body`, `replace_item`,
  `rename`, `add_function`, `edit_file`, `create_file`, `delete_file`.
- `run_cargo` - verify. Compile errors are classified as
  introduced-by-this-edit vs pre-existing so the model only chases
  its own breakage. Raw cargo NDJSON is dropped from the response;
  only parsed errors + a hint come back.
- `show_flow` - render a control-flow diagram for the human.

That's the whole surface. No grep, no shell, no arbitrary tool calls.

## Slash commands

- `/help` - list commands
- `/model [name]` - switch provider or model (fuzzy-searchable, sizes
  shown for Ollama variants)
- `/think on|off` - toggle reasoning visibility
- `/trace on|off` - dump raw stream chunks (debugging)
- `/flow <fn>` - open the control-flow diagram for a function
- `/skyline` - open the whole-workspace map
- `/resume` - pick a prior session and load it
- `/clear` - reset conversation (keeps system prompt + ontology)
- `/reindex` - re-walk the workspace
- `/warm` - force-reload the Ollama model
- `/export [path]` - dump the next-turn payload to JSON
- `/turns N` - set max tool-call rounds per user message

## Session persistence

Conversations save per workspace under
`~/.plane-code/sessions/<workspace-hash>/<id>.json`. The latest 30
sessions per workspace are kept; older ones are pruned automatically.
`/resume` lists them and reloads the chosen one onto the live agent
with on-screen replay.

## Development

```bash
git clone https://github.com/ahtavarasmus/plane-code
cd plane-code
cargo build --release
./target/release/plane-code
```

## License

Dual-licensed under MIT or Apache-2.0, at your option. See
`LICENSE-MIT` and `LICENSE-APACHE`.
