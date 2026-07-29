# lower.ch06 evidence provenance

- Fixture: `data/fixtures/lower-ch06.json`
- Binding: SHA-256 stored in `oracle.json`
- Origin: synthetic interarrival times and a hand-authored six-event limit-order-book replay.
- Random source: NumPy PCG64 with the declared seed; theoretical exponential moments remain independent of the simulated sample.
- No exchange feed, client order, latency measurement, or production execution claim is included.
