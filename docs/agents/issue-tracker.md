# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub issues. Use the `gh` CLI for all operations.

The intended remote is `Jason-W507/math-for-quant`. Until that remote exists, do not publish issues; retain approved design decisions in the active conversation or a user-approved local specification.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`.
- **Read an issue**: `gh issue view <number> --comments`, including labels.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments` with appropriate label and state filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`.
- **Apply or remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`.
- **Close**: `gh issue close <number> --comment "..."`.

Infer the repository from `git remote -v` after the GitHub remote is created.

## Pull requests as a triage surface

**PRs as a request surface: no.**

External pull requests are handled manually and are not pulled into the issue triage state machine.

## When a skill says "publish to the issue tracker"

Create a GitHub issue after the remote repository exists.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.
