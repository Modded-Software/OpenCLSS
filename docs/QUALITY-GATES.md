# Quality gates

`make check` runs three gates, and the native git hooks run the first two on
every commit:

| Target | Command | Hook |
| --- | --- | --- |
| `make lint` | `uv run ruff format --check .` then `uv run ruff check .` | `ci-lint` |
| `make type-check` | `uv run mypy .` | `ci-type-check` |
| `make test` | the build and the test programs | none |

Both tools are declared in the `dev` group of `pyproject.toml`, so the gate is
reproducible from `uv.lock`. Ruff and mypy each read exactly one configuration
file: ruff reads `ruff.toml` (it does not merge with `pyproject.toml`), and mypy
reads `[tool.mypy]` in `pyproject.toml` (it does not read `ruff.toml`).

## Why some rules are switched off here

`ruff.toml` selects the same rule catalogue as the workspace CI
(`/opt/workspace-ci/ruff.toml`). Several of those rules have real findings in
this tree, and they are turned off in the `ignore` list of `ruff.toml` so the
gate can go green while the findings are paid down one by one. This is a
deferral, not a decision that the rule is wrong: delete a code from the
`ignore` list, run `uv run ruff check .`, and read what is left.

The counts below are the state at the wiring commit.

| Rule | Findings | Why it is deferred |
| --- | --- | --- |
| `TRY003` | 350 | One custom exception class per failure mode, with no caller that branches on it. |
| `PLR2004` | 325 | The kernels are written in concrete tile, lane and byte sizes; naming each would not help. |
| `UP031` | 230 | Percent formatting is rewritten as each site is touched, not in one sweep. |
| `PLC0415` | 96 | `torch`, `cv2`, the Vulkan loader and the win32 API load inside functions so a host without them can still import the module. |
| `B023` | 91 | Closures that read the loop variable after the loop; each needs a look, not a rewrite. |
| `PLR0913` | 83 | One argument per packed operand by design. |
| `E402` | 67 | Module imports that must follow the environment setup above them. |
| `PLR0917` | 65 | Positional operands on the low-level entry points. |
| `C901` | 57 | Long numeric kernels. |
| `PLR0915` | 51 | Same. |
| `PLW1510` | 37 | `subprocess` calls whose failure the caller inspects itself. |
| `PLR0912` | 33 | Branchy format decoding. |
| `TRY301` | 29 | Raises inside a `try` that also does real work. |
| `E741` | 19 | Short names such as `l` in matrix code. |
| `PLW2901` | 16 | Loop name reused, in several cases on purpose. |
| `SIM115` | 15 | File handles held across a function body. |
| `PLW0603` | 11 | Module-level registries. |
| `B008` | 7 | Call in a default argument. |
| `PT028` `PT011` `PT018` `PT017` | 24 | The suites subclass `unittest.TestCase`. |
| `TRY300` `TRY004` | 13 | Exception style, as `TRY003`. |
| `PERF401` `RUF012` `PLR0911` `PLC3002` | 20 | Small, individually reviewed. |
| `SIM117` `SIM112` `B007` `SIM102` | 13 | Small. |
| `PYI024` `PLC0206` `F402` `RUF034` | 5 | Small, single sites. |
| `E501` | 73 | Retained at a 100-column limit; the remainder are long literals that cannot wrap. |

`mypy` runs in a deferred mode too: `strict` is not enabled, and the codes that
only fire because the tree is not annotated yet are listed in
`disable_error_code` (see `pyproject.toml`) with the same intent. Turning
`strict = true` on adds roughly 4000 `no-untyped-def` / `no-untyped-call`
findings, which is the annotation project, tracked separately.

## Paying a deferral down

1. Remove the code from the `ignore` list in `ruff.toml` (or from
   `disable_error_code` in `pyproject.toml`).
2. Run `uv run ruff check .` (or `uv run mypy .`) and work the findings.
3. Run `make check` to confirm the other gates still pass.