# Contributing

Thanks for contributing.

## Expectations

- Keep pull requests narrow and intentional.
- Prefer simple, explicit implementations over clever ones.
- Document behavior changes in the same pull request.
- Run the fastest relevant checks before opening or updating a pull request.
- Keep unrelated cleanup out of scope.

## Prerequisites for evals and examples

Model evals and local examples use Ollama.

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama (keep this running in a separate terminal)
ollama serve

# Pull models used by examples/evals
# examples/chat.py
ollama pull qwen3:1.7b
ollama pull bge-m3

# examples/custom_anchor.py (default model)
ollama pull qwen3:4b-instruct

# tests/evals fixtures
ollama pull qwen3:0.6b
```

For Linux and Windows install steps, use the official Ollama docs: <https://docs.ollama.com/>

If you only run deterministic checks (`-m "not eval"`), Ollama is optional.

To avoid leaving heavy model workloads running:

```bash
# Show currently loaded/running models
ollama ps

# Stop a loaded model (replace with a model shown by `ollama ps`)
ollama stop qwen3:4b-instruct
```

If you started the server with `ollama serve` in a terminal, stop it with `Ctrl+C` in that same terminal.
If you started Ollama as a background service, stop it with your service manager (for example `brew services stop ollama` on Homebrew installs).

## Local workflow

At minimum, contributors should run the fastest checks available for the area they changed.

Recommended setup and baseline checks:

```bash
# Install project + dev tooling
uv sync --group dev

# Fork the repository on GitHub, then clone your fork.
# Create a feature branch in your fork before running full hooks
git checkout -b yourname/short-topic

# Install git hooks
uv run pre-commit install

# Run all configured hooks
uv run pre-commit run --all-files

# Run deterministic tests (same policy as CI)
uv run pytest -m "not eval"

# Run model evals manually when needed
uv run pytest -m eval tests/evals
```

Examples of targeted checks:

- formatter checks
- linter checks
- type checks
- focused tests
- `git diff --check` for docs and text-heavy changes

If the repository later adds project-specific commands, document them here and in the README.

## Pull request rules

- Use Conventional Commit-style titles such as `feat:`, `fix:`, `docs:`, `ci:`, or `chore:`.
- Link the canonical issue or related pull request in the PR template.
- Fill out the pull request checklist completely.
- Explain tradeoffs or follow-up work in the PR body when relevant.
- Expect an automatically requested reviewer once the pull request is ready for review.

## Issue and PR intake

- Search existing issues and pull requests before opening a new one.
- Pick the closest matching `Area` in the issue form so automation can route triage correctly.
- New issues are assigned automatically from the checked-in maintainer rotation.
- Maintainers can mark an issue as claimable with `/triaged` (adds `status: ready-for-work`).
- Contributors can claim work with `/claim` when an issue is ready.
- Contributors can hold one claimed issue at a time.
- Claimed issues move to `status: in-progress`.
- Contributors can release their own claim with `/unclaim`.
- If a claimed issue has no claimer activity for 7 days, automation posts a follow-up.
- If there is still no claimer response 24 hours after the follow-up, automation unclaims the issue and returns it to `status: ready-for-work`.
- Maintainers can close duplicates with `/duplicate #123`.
- Maintainers can close superseded pull requests with `/superseded-by #123`.

## Communication

- Bug reports, feature requests, and project tasks should use the checked-in issue forms.
- Security issues should follow [SECURITY.md](SECURITY.md).
- Significant project direction changes should be reflected in [docs/vision.md](docs/vision.md) or [docs/roadmap.md](docs/roadmap.md).
