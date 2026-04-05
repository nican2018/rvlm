# Contributing

Thank you for your interest in contributing to RVLM!

## Guidelines

- Avoid modifying files under `rvlm/core/` unless necessary — the core loop is kept minimal on purpose.
- Run `ruff check --fix .` and `ruff format .` before submitting a PR.
- Include unit tests for new features where practical.

## Roadmap

### High Priority
- [ ] **Additional sandbox backends** — new sandboxed execution environments.
- [ ] **Persistent REPL across turns** — flag to keep REPL state between `completion()` calls.
- [ ] **Additional benchmarks / examples** to demonstrate usage.
- [ ] **Improve documentation** — API docs, tutorials.
- [ ] **Unit tests** — more comprehensive test coverage with each PR.

### Nice to Have
- [ ] **Arbitrary input modalities** — generalise beyond `str` / image to any picklable input.
- [ ] **File-system + bash environments** — beyond REPLs.
- [ ] **Improved trajectory visualiser** — richer UI, filtering, comparisons.

### Research Directions
- [ ] **Pipelining / async LM calls** — overlap generation with execution.
- [ ] **Efficient prefix caching** — avoid redundant KV-cache computation.
- [ ] **Training models as RLMs** — see the [verifiers rvlm_env](https://github.com/PrimeIntellect-ai/verifiers/blob/main/verifiers/envs/experimental/rvlm_env.py).


