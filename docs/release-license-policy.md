# Release license and provenance policy

The release manifest records the version, Git commit, source date epoch, byte size, SHA-256 digest, artifact kind, and license identifier for every published PDF, executed notebook, notebook archive, synthetic data file, and vendored template asset.

| Material | License | Authority |
| --- | --- | --- |
| Original Python source | MIT | `LICENSE` |
| Original manuscript and explanatory notebook cells | CC BY-NC-SA 4.0 | `LICENSE-CONTENT.md` |
| Executed notebooks | MIT and CC BY-NC-SA 4.0, according to cell content | `LICENSE`, `LICENSE-CONTENT.md` |
| Original synthetic fixtures and chapter 17 synthetic rows | CC0 1.0 | `data/README.md`, `data/ch17/README.md` |
| ElegantBook class and copied template cover | LPPL 1.3c | `vendor/elegantbook/LPPL-License.txt`, `docs/template-provenance.json` |

No real market dataset is shipped. A future external dataset must add its own retrieval date, version, covered dates, checksum, transformation record, and license before it may enter a release.
