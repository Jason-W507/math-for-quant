# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- `CONTEXT.md` at the repository root.
- Relevant ADRs under `docs/adr/`.

If either is absent, proceed silently. Domain-modeling workflows create them when terminology or durable decisions are actually resolved.

## File structure

This repository uses a single context:

```text
/
├── CONTEXT.md
├── docs/adr/
└── ...
```

The common mathematical volume and direction-specific volume share one glossary, prerequisite graph, notation policy, exercise system, and publication pipeline.

## Use the glossary's vocabulary

Use terms as defined in `CONTEXT.md` when naming issues, hypotheses, tests, chapter concepts, and mathematical objects. Avoid drifting between synonyms without recording the distinction.

## Flag ADR conflicts

If proposed work contradicts an existing ADR, surface the conflict explicitly rather than silently overriding the decision.
