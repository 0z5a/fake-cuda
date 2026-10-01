# Short-trace CPU optimization and whole-model validation

The short selection trace was dominated by report serialization rather than virtual device work. A 4000-operator CPU profile spent 0.537 s in report digests, including 0.447 s in recursive `dataclasses.asdict` conversion, out of 0.971 s total instrumented execution. The report now passes dataclass fields directly to the JSON encoder. Canonical bytes, SHA-256 digests, virtual timelines and accounting remain identical. These instrumented times identify the bottleneck; the table below uses separate, unprofiled runs.

## Equal-target complete process comparison

Both versions run offline with the same frozen stage profile and held-out trace. The baseline is `b6637b3`; the candidate changes only report serialization. Five independent APPA/PAAP sessions launch 20 fresh processes per trace size, ten per version. Wall time includes interpreter startup, imports, profile/trace loading, replay, hashing, actor delivery, output and natural teardown. Native calibration and profile fitting are excluded equally from both arms. CUDA is hidden in every CPU process.

| Operators per process | Before, median s | After, median s | CPU speedup | Wall-time reduction | Session speedup range |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 400 | 0.091140 | 0.079732 | 1.143× | 12.52% | 1.116–1.179× |
| 4000 | 0.376063 | 0.267963 | 1.403× | 28.75% | 1.374–1.463× |

All 40 processes preserve their corresponding timeline/ledger/reply digests. Every session improves at both sizes. This comparison measures the source change; the earlier [paced/offline comparison](moe-serving.md) measures a different change and cannot be substituted for this baseline. Neither comparison measures numerical neural-network acceleration.

| Evidence | SHA-256 |
| --- | --- |
| Frozen profile | `2dfcc29b95016ecb5c42fc20e3a6da08a92e200bab003a23df4cb543564c8d08` |
| Held-out trace | `47ccea2b9608bbcd1135753819398ab709d09104e2f87c403c5fb46c58aecd9f` |
| 400-operator target | `e687baa09f7059ba0612dd232d72db03c602aa017d321a8365640a0dbe876eef` |
| 4000-operator target | `45743e2352b04101e0fc2fe78d49c8cc7b6ce9f52306229f6aaf3a62dd1e8476` |

The canonical-byte regression covers MoE, KDA, selection and unknown operators across fresh execution IDs. All 13 Python semantic cases and all 34 registered contracts pass. Raw profiles, timings, logs and JSON stay outside Git.
