# Lower Chapter 1 provenance

The panel is synthetic and hand-designed. Each future return equals a stated intercept plus a known cross-sectional slope times the signal. The two slopes, long-short returns, cost deductions and p-value decisions can therefore be checked without calling the implementation. No market data or production claim is involved.

The 2016--2024 protocol panel is generated once from NumPy PCG64 seed 41 with 108 observations, horizon strengths 0.5, 0.3 and 0.1, and noise scale 0.5. Canonical JSON containing the signal and three return arrays has SHA-256 `2e54185ee940888b9864d5728f41d5d7c2b35ea5961804ddb66dc3c14020d2ce`. The independent test reconstructs that frozen panel, checks the hash, and derives Pearson correlations with a plain-Python mean/covariance formula rather than calling the production `protocol_metrics` or `correlation` functions.
