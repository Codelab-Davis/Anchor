# Maintainer Setup

This repository relies on several GitHub settings that cannot be expressed entirely through checked-in files.

## Day 0 checklist

Before accepting external contributions, complete these setup steps:

- verify project metadata in `README.md` and `SECURITY.md` is current
- verify handles in `.github/CODEOWNERS` and `.github/maintainers.yml` are current
- verify `.github/dependabot.yml` covers the project stack
- push the default branch once so GitHub Actions and label sync can initialize
- verify that new issues receive an `area:*` label and assignee
- verify that new pull requests receive labels and a requested reviewer

## Branch protection

Protect `main` with these required checks once the workflows are live:

- `Docs / markdown`
- `Docs / links`
- `PR Hygiene / title-and-checklist`

Require at least one approving review and block force-pushes.

## Repository settings to enable

- secret scanning
- push protection
- Dependabot alerts
- Dependabot security updates
- GitHub Actions enabled for this repository
- Discussions, if the project wants community Q&A outside issues

## Maintainer routing

The checked-in maintainer routing file lives at `.github/maintainers.yml`.

- `triage_assignees`: GitHub handles eligible for automatic issue assignment
- `pr_reviewers`: GitHub handles eligible for automatic pull request review requests

Keep the lists small and current. The workflows use deterministic round-robin selection based on the issue or pull request number and skip the author when possible.

## Maintainer triage commands

Maintainers can manage issue and pull request flow directly from comments:

- `/triaged`: remove early triage labels after initial review
- `/needs-info`: mark the item as waiting on reporter or author follow-up
- `/duplicate #123`: close an issue or pull request in favor of the canonical item
- `/superseded-by #123`: close a pull request in favor of a newer replacement pull request

If the project later adds custom triage command automation, document it here.

## Intake automation

- `Issue Intake` checks new and edited issues for form completeness, applies the selected `area:*` label, assigns a maintainer, and suggests similar existing work.
- `PR Hygiene` checks pull request titles, checklist compliance, linked context, and similar existing pull requests.
- `PR Routing` requests a reviewer from the checked-in maintainer rotation when a pull request is ready for review.
- `Welcome` posts a first-time contributor note on the first issue and first pull request in the repository.
- Automation-owned labels use the `needs:` prefix so maintainers can distinguish bot findings from final `status:` labels.

## Labels

The canonical label set lives in `.github/labels.json`. Update that file and let the `Label Sync` workflow apply the changes.

## Dependabot

This repository currently enables Dependabot updates for `uv` and GitHub Actions.
Add additional ecosystems in `.github/dependabot.yml` if the stack expands.

## Optional later additions

This base template intentionally does not include GitHub Project automation or release bots. Add those after the repository has stable maintainer ownership and a real release process.
