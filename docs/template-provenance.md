# ElegantBook template provenance

- Source directory: external, read-only `D:\Latex\ElegantBook`
- Template version: ElegantBook 4.7 (2026-05-01)
- Copied class SHA-256 before compatibility patch: `D2CDB802B3DE46B1D659D1A8EB36979AECD761402D3E95296936F433F97549EB`
- Copied LPPL license SHA-256: `5F05FCF6EF25A6C31BCCD2DF7C0C46B23107BBEB2CE5CDBA74EFB5CC357F4DBB`
- Copied cover image SHA-256: `0748354F5D61633F9032DAB0A4A6774CB91CF1A0FC5892CC52D73A47A4552A0B`

MiKTeX 26.1 can locate `adforn.sty`, but that package requires the unavailable `fixtounicode.sty` when automatic installation is disabled. The vendored class therefore removes only the `adforn` requirement and replaces the two problem-set flourishes with `pifont`'s already-loaded `\ding{166}`. The external template directory is not modified.
