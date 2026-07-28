# Editorial style

## Separate mathematical and research roles

Core exposition follows three steps: state the mathematical object and result, give intuition or a worked example, then explain the quant-research consequence. Research commentary does not interrupt a proof.

Use the four reader-facing categories defined in `tex/common/evidence-contract.tex`:

- theorem and proof;
- computational check;
- failure boundary;
- research connection.

The global evidence contract is explained once in the front matter. Chapters must not repeat generic claims such as “code does not prove a theorem” or “a successful run does not establish research validity”; they should state only the chapter-specific oracle, failure, assumption, or research limitation.

Use `heuristic`, `diagnostic`, and `implementationnote` for approximations, diagnostic formulas, and implementation-specific observations. Do not visually present them as theorems.

## Terminology

- Write `\(\sigma\)-代数` and `\(L^p\) 空间` in Chinese prose.
- Introduce bilingual terms once, then use the chosen Chinese term consistently.
- Reserve “oracle” for an independently derived computational reference.
- Keep page counts out of durable chapter contracts; they are publication diagnostics only.
