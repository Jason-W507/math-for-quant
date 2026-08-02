# Release license and provenance policy

The release manifest records the version, Git commit, source date epoch, byte size, SHA-256 digest, artifact kind, source, license identifier, and license-file path for every published PDF, tracked teaching notebook, notebook archive, registered data file, and vendored template asset. The notebook archive carries the MIT text, the manuscript license, and this mapping policy; publishing copies the tracked notebooks and does not create a second executed source tree.

| Material | License | Authority |
| --- | --- | --- |
| Original Python source | MIT | `LICENSE` |
| Original manuscript and explanatory notebook cells | CC BY-NC-SA 4.0 | `LICENSE-CONTENT.md` |
| Tracked teaching notebooks | MIT and CC BY-NC-SA 4.0, according to cell content | `LICENSE`, `LICENSE-CONTENT.md` |
| Original synthetic fixtures and chapter 17 synthetic rows | CC0 1.0 | `data/README.md`, `data/ch17/README.md` |
| Frozen public-data teaching snapshots | Source-specific public-domain or attribution terms | `data/assets.json`, `data/real/*` provenance notes |
| ElegantBook class and copied template cover | LPPL 1.3c | `vendor/elegantbook/LPPL-License.txt`, `docs/template-provenance.json` |

`data/assets.json` is the exhaustive release registry. Any unregistered file under `data/` blocks publication. Small frozen public-data snapshots may be shipped only as provenance and input-protocol teaching evidence; they are not live feeds, tradable quote histories, or performance evidence. Every external snapshot must record retrieval date, version, covered dates, field schema, row/column selection, missing-value handling, transformations, checksum and license before it may enter a release.
