# ElegantBook template provenance

- Source directory: external, read-only `D:\Latex\ElegantBook`
- Template version: ElegantBook 4.7 (2026-05-01)
- Machine-readable baseline: `docs/template-provenance.json`

MiKTeX 26.1 can locate `adforn.sty`, but that package requires the unavailable `fixtounicode.sty` when automatic installation is disabled. The vendored class therefore removes only the `adforn` requirement and replaces the two problem-set flourishes with `pifont`'s already-loaded `\ding{166}`. The external template directory is not modified.
