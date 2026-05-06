# DataTalks.Club FAQ Benchmark Plan

Date: 2026-05-04

Source repository: https://github.com/DataTalksClub/faq

## Why This Dataset Fits

The FAQ repository is a good project-owned benchmark for `minsearch` because it
is close to a real user workflow:

- Users ask natural troubleshooting questions.
- Documents are short FAQ entries with a question and answer.
- The FAQ automation already uses `minsearch.Index` over `section`, `question`,
  and `answer`.
- The corpus is large enough to expose ranking issues, but small enough to
  inspect manually.

Current inspected size:

| Course | FAQ documents | Sections |
|---|---:|---:|
| `data-engineering-zoomcamp` | 393 | 15 |
| `llm-zoomcamp` | 79 | 9 |
| `machine-learning-zoomcamp` | 433 | 19 |
| `mlops-zoomcamp` | 249 | 8 |
| Total | 1,154 | 51 |

FAQ markdown length distribution by word count:

| min | median | max |
|---:|---:|---:|
| 14 | 88 | 693 |

This size is practical for fast local benchmarks and for manual pooled judging.

## Important Caveat

The FAQ corpus does not automatically give complete relevance judgments.

Each FAQ file has a `question`, so we can treat that question as a query and the
same FAQ document as relevant. That gives a useful **known-item retrieval**
benchmark:

- Did the search engine retrieve the exact FAQ entry?
- How high was it ranked?

But this does **not** give complete precision, because other FAQ entries may also
be relevant and are unlabeled. For true precision/nDCG, we need manual qrels.

So the benchmark should have two layers:

1. Automatic known-item retrieval over all FAQ questions.
2. Manually labeled pooled relevance judgments for realistic user queries.

## Layer 1: Automatic Known-Item Benchmark

Use every FAQ question as a query.

Document format:

```python
{
    "course": "data-engineering-zoomcamp",
    "section": "Module 1: Docker",
    "section_id": "module-1-docker",
    "question": "Docker: Cannot connect to Docker daemon ...",
    "answer": "...",
    "document_id": "5e6c4090af",
}
```

Index fields:

```python
text_fields = ["section", "question", "answer"]
keyword_fields = ["course", "section_id", "document_id"]
```

Queries:

- `question`
- optionally `question + "\n\n" + answer[:500]` for FAQ proposal triage mode

Relevant document:

- the same `document_id`

Metrics:

- `Hit@1`, `Hit@3`, `Hit@5`, `Hit@10`
- `MRR@10`
- source document rank distribution

Recommended variants:

| Variant | Why |
|---|---|
| all courses, no filter | Tests ambiguity across the whole FAQ corpus. |
| course-filtered | Matches FAQ automation when course is known. |
| question-only query | Search site behavior. |
| question + answer query | FAQ proposal triage behavior. |
| `Index` vs `AppendableIndex` | Checks ranking parity. |

Expected interpretation:

- This is mostly a recall/rank benchmark.
- It should be cheap enough to run in CI.
- It should catch regressions where obvious exact FAQ questions stop retrieving
  their own entries.

## Layer 2: Manual Pooled Relevance Benchmark

This is the benchmark for actual precision and nDCG.

Create 100-200 realistic user queries, sampled across courses and sections.

Examples:

```yaml
- query_id: de_docker_daemon
  course: data-engineering-zoomcamp
  query: cannot connect to docker daemon is docker running

- query_id: de_pgcli_port
  course: data-engineering-zoomcamp
  query: pgcli cannot connect to postgres on port 5432

- query_id: ml_certificate
  course: machine-learning-zoomcamp
  query: how do I get my certificate

- query_id: mlops_project_eval
  course: mlops-zoomcamp
  query: how is the capstone project evaluated
```

For each query:

1. Run multiple retrieval configurations.
2. Pool the top 20 results from each configuration.
3. Manually label pooled documents:
   - `2 = highly relevant`
   - `1 = partially relevant`
   - `0 = not relevant`

Configurations to pool:

- `Index`
- `AppendableIndex`
- question/title boosted config
- answer boosted config
- section boosted config
- later: BM25 baseline if added

Metrics:

- `Precision@1`, `Precision@3`, `Precision@5`
- `Recall@5`, `Recall@10`
- `MRR@10`
- `nDCG@5`, `nDCG@10`
- bad top result rate: rank 1 has relevance `0`

This gives the proper precision/recall benchmark we need for ranking changes.

## Layer 3: Hard Regression Set

Keep a small YAML file of handpicked failure cases. These should be easy to
inspect and should fail loudly.

Examples:

- Query has short common terms.
- Query contains tool names shared across many courses.
- Query is about Docker but retrieves unrelated WSL/GCP docs.
- Query is about dashboards or reports and retrieves unrelated “judge” or
  generic docs.
- Query has typo or variant wording.

Metrics:

- expected document must be in top 5
- rank 1 must not be from a known-bad section
- rank 1 must have relevance > 0 if manual qrels exist

## Suggested Files

```text
benchmark/relevance/
  datasets.py
  metrics.py
  run_faq_known_item.py
  run_faq_qrels.py
  faq_queries.yaml
  faq_qrels.yaml
  README.md
```

The FAQ repository should not be vendored into `minsearch`. The runner can accept
one of:

- `--faq-dir /path/to/faq`
- `--clone-faq` to clone into a temp/cache directory
- `FAQ_DIR=/path/to/faq`

## First Implementation Step

Implement the known-item benchmark first.

It requires no manual labeling and will immediately tell us:

- exact FAQ question retrieval quality
- how much course filtering helps
- whether `AppendableIndex` matches `Index`
- which FAQ questions are hard for lexical search

Then use the failure list from that run to seed the manual qrels benchmark.

## Why Not Only Use This Dataset

The FAQ benchmark should be project-owned, but not the only benchmark.

Keep public datasets such as Cranfield, NFCorpus, and SciFact as general IR
guardrails. Then use the FAQ benchmark to catch real docs/search behavior that
public datasets do not represent.

