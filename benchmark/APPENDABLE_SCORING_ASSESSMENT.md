# AppendableIndex Scoring Assessment

Date: 2026-05-04

## Goal

Assess whether the `AppendableIndex` scoring fix behaves sensibly compared with
the regular `Index`.

The assessment checks:

- Regression behavior for long irrelevant documents.
- Tokenization behavior for one-character query terms such as `I` and `a`.
- `fit()` versus `append()` ranking consistency.
- Real documentation search behavior on the Evidently docs corpus.
- Precision/recall-style IR metrics on an open benchmark dataset with qrels.

`gitsource` and IR benchmark tooling were installed only in
`/tmp/minsearch-repro`; they were not added as project dependencies.

## Open Dataset

For precision/recall, I used the Cranfield collection through `ir_datasets`.

Sources:

- `ir_datasets` catalog: https://ir-datasets.com/
- Cranfield dataset page: https://ir-datasets.com/cranfield.html

Cranfield is a small standard IR test collection:

- Documents: 1,400 scientific abstracts
- Queries: 225 natural-language queries
- Qrels: 1,837 relevance judgments

This is a better fit for precision/recall than ad hoc docs because it provides
ground-truth relevance assessments.

## Method

For local synthetic and Evidently docs checks:

- Build `Index` and `AppendableIndex` with the same fields.
- Run the same queries against both.
- Compare top-5 ranking order and whether the expected document appears first.

For Cranfield:

- Indexed `title` and `text`.
- Used document ID as a keyword field.
- Ran all 225 queries.
- Collected top 50 results from both index implementations.
- Computed `P@k`, `R@k`, `MRR@k`, and `nDCG@k` for `k = 5, 10, 20, 50`.

## Synthetic Checks

### Long Irrelevant Content

Dataset shape:

- `dashboard-guide`: concise document matching both `create` and `dashboard`.
- `dashboard-noisy`: very long document repeating only `dashboard`.
- `create-noisy`: very long document repeating only `create`.

Results:

| Query | Index top results | AppendableIndex top results | Assessment |
|---|---|---|---|
| `create dashboard` | `dashboard-guide`, `create-noisy`, `dashboard-noisy` | `dashboard-guide`, `create-noisy`, `dashboard-noisy` | Good. Concise full match ranks first. |
| `dashboard` | `dashboard-guide`, `dashboard-noisy` | `dashboard-guide`, `dashboard-noisy` | Matches `Index`. |
| `create resources` | `create-noisy`, `dashboard-guide` | `create-noisy`, `dashboard-guide` | Matches `Index`. |

This directly tests the bug class where scoring only over query-token slices can
over-rank noisy documents. The fixed behavior matches `Index`.

### One-Character Query Noise

Dataset shape:

- `dashboard`: actual dashboard creation document.
- `letter-noise`: contains many `i` and `a` tokens plus unrelated text.
- `input-data`: relevant to creating data, not dashboards.

Results:

| Query | Index top results | AppendableIndex top results | Assessment |
|---|---|---|---|
| `how do I create a dashboard?` | `dashboard`, `input-data`, `letter-noise` | `dashboard`, `input-data`, `letter-noise` | Good. Letter noise does not outrank dashboard. |
| `I A` | no results | no results | Matches sklearn-style token length behavior. |
| `create data` | `input-data`, `dashboard`, `letter-noise` | `input-data`, `dashboard`, `letter-noise` | Matches `Index`. |

This confirms the tokenizer change is important: by default, `AppendableIndex`
now ignores one-character tokens just like sklearn's default `TfidfVectorizer`.

### Field Ranking

Dataset shape:

- `title-hit`: match in title only.
- `content-hit`: repeated content match.
- `full-hit`: title, description, and content match.

Results:

| Query | Index top results | AppendableIndex top results | Assessment |
|---|---|---|---|
| `dashboard panel` | `full-hit`, `title-hit`, `content-hit` | `full-hit`, `title-hit`, `content-hit` | Matches `Index`. |
| `create dashboard` | `full-hit`, `title-hit`, `content-hit` | `full-hit`, `title-hit`, `content-hit` | Matches `Index`. |

### Append Sequence Versus Full Fit

The same four synthetic documents were indexed in two ways:

- regular `Index.fit(all_docs)`
- `AppendableIndex.fit(initial_docs)` followed by `append()` for remaining docs

Results:

| Query | Index top results | AppendableIndex top results | Assessment |
|---|---|---|---|
| `create dashboard` | `dashboard-guide`, `create-noisy`, `dashboard-noisy` | `dashboard-guide`, `create-noisy`, `dashboard-noisy` | Matches. |
| `create resources` | `create-noisy`, `dashboard-guide` | `create-noisy`, `dashboard-guide` | Matches. |
| `dashboard` | `dashboard-guide`, `dashboard-noisy` | `dashboard-guide`, `dashboard-noisy` | Matches. |

This suggests recomputing IDF and full document norms after appends is working.

## Evidently Docs Repro

Setup:

```python
from gitsource import GithubRepositoryDataReader, chunk_documents

reader = GithubRepositoryDataReader(
    repo_owner="evidentlyai",
    repo_name="docs",
    allowed_extensions={"md", "mdx"},
)
files = reader.read()
parsed_docs = [doc.parse() for doc in files]
chunked_docs = chunk_documents(parsed_docs, size=3000, step=1500)
```

Fields:

- text: `title`, `description`, `content`
- keyword: `filename`

Top-5 results:

| Query | Index | AppendableIndex | Assessment |
|---|---|---|---|
| `how do I create a dashboard?` | dashboard API, synthetic inputs, dashboard UI, dashboard overview, dashboard API | same | Fixed original issue. |
| `create dashboard` | dashboard API, dashboard overview, dashboard API, dashboard UI, dashboard API | same | Good. |
| `add dashboard panels` | dashboard API, dashboard UI, dashboard API, dashboard API, dashboard UI | same | Good. |
| `how to add monitoring panel` | tags metadata, dashboard panel types x4 | same | Same as `Index`; relevance is debatable but parity is good. |
| `synthetic input data` | datasets generate, synthetic data API, why synthetic, introduction, input data | same | Good. |
| `LLM as a judge` | LLM judge chunks | same | Good. |
| `data drift report` | report, report, report, data drift, data drift | report, report, data drift, report, data drift | Minor chunk-order difference. |

For `data drift report`, the difference is a swap among highly similar relevant
chunks:

```text
Index
  57:docs/library/report.mdx
  58:docs/library/report.mdx
  59:docs/library/report.mdx
  354:metrics/preset_data_drift.mdx
  356:metrics/preset_data_drift.mdx

AppendableIndex
  57:docs/library/report.mdx
  58:docs/library/report.mdx
  354:metrics/preset_data_drift.mdx
  59:docs/library/report.mdx
  356:metrics/preset_data_drift.mdx
```

That looks acceptable: all top chunks are on-topic and the first two are
identical.

## Cranfield Metrics

| Index | k | P@k | R@k | MRR@k | nDCG@k |
|---|---:|---:|---:|---:|---:|
| Index | 5 | 0.2862 | 0.2507 | 0.4938 | 0.2369 |
| AppendableIndex | 5 | 0.2853 | 0.2496 | 0.4956 | 0.2359 |
| Index | 10 | 0.2049 | 0.3444 | 0.5057 | 0.2615 |
| AppendableIndex | 10 | 0.2058 | 0.3458 | 0.5078 | 0.2623 |
| Index | 20 | 0.1407 | 0.4490 | 0.5117 | 0.2993 |
| AppendableIndex | 20 | 0.1413 | 0.4510 | 0.5129 | 0.3009 |
| Index | 50 | 0.0772 | 0.5804 | 0.5131 | 0.3438 |
| AppendableIndex | 50 | 0.0774 | 0.5828 | 0.5143 | 0.3446 |

Assessment:

- Aggregate quality is effectively the same.
- `AppendableIndex` is very slightly higher at `k >= 10` on this run.
- Differences are small enough to be explained by near-ties and tokenizer/scorer
  implementation details.

## Cranfield Ranking Parity

| k | Average top-k overlap | Exact same ordered top-k rate |
|---:|---:|---:|
| 5 | 0.9796 | 0.8044 |
| 10 | 0.9831 | 0.5600 |
| 20 | 0.9849 | 0.2711 |

Interpretation:

- The two implementations retrieve almost the same documents.
- Exact ordering naturally diverges more as `k` grows because lower-ranked
  documents have closer scores.
- The metric table above shows those ordering differences do not hurt aggregate
  retrieval quality.

Example top-5 disagreements:

```text
qid 8
query: what methods -dash exact or approximate -dash are presently available for predicting body pressures
Index:       492, 461, 1082, 232, 711       relevant hits: F F F F F
Appendable: 492, 461, 1082, 232, 48        relevant hits: F F F F T

qid 34
query: have wind tunnel interference effects been investigated on a systematic basis
Index:       672, 610, 799, 431, 516        relevant hits: T F T T F
Appendable: 672, 610, 516, 799, 431        relevant hits: T F F T T
```

These examples are small ordering swaps, not obvious relevance regressions.

## Conclusion

The scoring behavior now looks sensible.

The fix addresses the observed failure mode:

- Long documents that match only part of a query are penalized by the full TF-IDF
  document norm.
- One-character query terms no longer lift unrelated documents by default.
- `fit()` and `append()` paths remain consistent after IDF/norm recomputation.

On the Cranfield benchmark, `AppendableIndex` has essentially the same
precision, recall, MRR, and nDCG as `Index`. On the Evidently docs repro, the
problem query now returns the same top-5 chunk IDs as `Index`.

Residual differences are expected because `AppendableIndex` is a separate
Python implementation of TF-IDF ranking rather than sklearn's exact matrix
pipeline. The remaining observed differences are small ranking swaps among
similar documents, not evidence of the previous irrelevant-result failure.

