# Data policy

SPDX-License-Identifier: CC0-1.0

All JSON files under `data/fixtures/` are original synthetic teaching data released under CC0 1.0. Frozen public-data snapshots under `data/real/` retain their source-specific provenance and license notes. `data/assets.json` is the exhaustive registry used by the release gate.

Future real-world slices must record source, retrieval or snapshot date, covered dates, schema, transformations, checksum and license. Synthetic data should be generated from documented parameters and must not be presented as market observations. A real dataset may illustrate data quality, but it cannot be the sole correctness oracle.
