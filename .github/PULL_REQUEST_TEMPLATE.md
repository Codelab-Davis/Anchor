# Pull Request

Use a Conventional Commit title: `feat: ...`, `fix: ...`, `docs: ...`, `chore: ...`, `refactor: ...`, `test: ...`, `build: ...`, `ci: ...`, `perf: ...`, or `revert: ...`.

## Summary

<!-- REQUIRED -->
<!-- 1-3 sentences describing what changed and why -->

## Linked context

<!-- REQUIRED -->
<!-- Keep this heading name: automation reads it -->
Closes # or Related to # or Supersedes #

## PR Size

<!-- REQUIRED: check one -->

- [ ] Small (< 100 LOC delta)
- [ ] Medium (100-499 LOC delta)
- [ ] Large (500-999 LOC delta)
- [ ] XL (1000+ LOC delta — must have been pre-discussed in an issue)

## PR Type

<!-- Check all that apply -->

- [ ] Bug fix
- [ ] New feature
- [ ] Prompt change (`prompt` label required; eval results required below)
- [ ] Test
- [ ] Docs
- [ ] Refactor / chore

## Context / Motivation

<!-- REQUIRED -->
<!-- Why is this change needed? -->

## Changes

<!-- REQUIRED -->
<!-- One line for small PRs; bullets for medium/large PRs -->

## How to test

<!-- Optional for small PRs; recommended for all -->

1.
2.

## Prompt Change Eval Results

<!-- Required if PR Type includes "Prompt change"; skip otherwise -->
<!-- Run: uv run pytest -m eval tests/evals -->
<!-- Preferred model: llama3:8b via Ollama -->

<details>
<summary>Full eval output (optional)</summary>

```text
<paste here>
```

</details>

## Checklist

- [ ] I ran the relevant local checks.
- [ ] I updated documentation or confirmed no docs changes were needed.
- [ ] I linked related issues or pull requests and confirmed this is not duplicate work.
- [ ] This change stays within the stated scope of the PR.

## Notes for reviewers

Call out any tradeoffs, follow-up work, or setup needed to validate the change.
