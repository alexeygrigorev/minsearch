# Benchmark Writeup: Index vs AppendableIndex Performance

## Overview

This document details the complete benchmarking process comparing the regular `Index` (based on sklearn's TfidfVectorizer) against the optimized `AppendableIndex` implementation. The goal was to measure performance differences in both indexing and search operations.

## Background

The `AppendableIndex` was created to support incremental document addition (append operations), which the regular `Index` doesn't support. The initial implementation was significantly slower, so we implemented several optimizations to bring performance closer to the regular index.

## Dataset

We used the Simple English Wikipedia dump for realistic benchmarking:

| Metric | Value |
|--------|-------|
| Source | https://dumps.wikimedia.org/simplewiki/ |
| Compressed Size | 444 MB (.xml.bz2) |
| Uncompressed Size | ~1 GB (JSONL) |
| Total Documents | 291,737 articles |
| Total Text Size | ~0.95 GB |

## Optimization Journey

### Phase 1: Initial Optimizations

The original AppendableIndex had significant performance issues:

Problem 1: Re-tokenization
- Tokenized documents during every search
- Fix: Cache tokenized documents per field

Problem 2: Duplicates in inverted index
- Used lists with duplicate document IDs
- Fix: Use sets for O(1) deduplication

Problem 3: On-the-fly IDF computation
- Computed IDF for every token during search
- Fix: Pre-compute IDF values after fit/append

### Phase 2: Search Performance Optimization

Problem: Counter() call during search
- Created Counter objects for every document during scoring
- Fix: Pre-compute token counts during indexing

## Benchmark Scripts

### Local Testing (1,000 documents)

```python
# benchmark/run_benchmark.py
- Load 1,000 documents from Wikipedia subset
- Index both implementations
- Run 10 random queries
- Report timing and QPS
```

### Full Wikipedia Benchmark

```python
# benchmark/run_full_benchmark.py
- Load all Wikipedia documents (291K)
- Index both implementations
- Run 10 random queries
- Report timing and QPS
```

## Benchmark Results

### Local Results (1,000 documents)

| Metric | Regular Index | AppendableIndex | Ratio |
|--------|---------------|-----------------|-------|
| Indexing Time | 2.6s | 2.95s | 1.13x slower |
| Search Time | 27.8ms | 1.42ms | 0.05x (20x faster) |
| QPS | 36.0 | 704.2 | 19.6x higher |

### AWS EC2 Results (t3.large, eu-west-1)

#### 10,000 Documents

| Metric | Regular Index | AppendableIndex | Ratio |
|--------|---------------|-----------------|-------|
| Indexing Time | 8.83s | 10.45s | 1.18x slower |
| Search Time | 127.61ms | 2.05ms | 0.02x (50x faster) |

#### 50,000 Documents

| Metric | Regular Index | AppendableIndex | Ratio |
|--------|---------------|-----------------|-------|
| Indexing Time | 24.13s | 33.63s | 1.39x slower |

#### 100,000 Documents

| Metric | Regular Index | AppendableIndex | Ratio |
|--------|---------------|-----------------|-------|
| Indexing Time | 47.91s | 64.13s | 1.34x slower |

#### 125,000 Documents (max for 8GB RAM)

| Metric | Regular Index | AppendableIndex | Ratio |
|--------|---------------|-----------------|-------|
| Indexing Time | 57.76s | 79.08s | 1.37x slower |
| Search Time | 1043.73ms (1.0 QPS) | 13.66ms (73 QPS) | 0.01x (76x faster) |

## Key Findings

### 1. Indexing Overhead
- Consistent 1.3-1.4x slower across all dataset sizes
- This is an acceptable trade-off for appendability
- Overhead comes from:
  - Building additional data structures (doc_tokens, doc_token_counts)
  - Set operations instead of list append
  - Pre-computation of IDF values

### 2. Search Performance
- Dramatically faster than regular index
- Performance advantage scales with dataset size:
  - 10K docs: 50x faster
  - 125K docs: 76x faster
- Reasons for speedup:
  - No re-tokenization needed (cached)
  - Pre-computed IDF values
  - Pre-computed token counts
  - Set-based document lookup

### 3. Memory Requirements

| Dataset Size | RAM Required |
|--------------|--------------|
| 1,000 docs | ~500 MB |
| 50,000 docs | ~2 GB |
| 100,000 docs | ~4 GB |
| 125,000 docs | ~6 GB |
| 291,000 docs (full) | ~12-14 GB |

The AppendableIndex uses more memory due to:
- Cached tokenized documents
- Pre-computed token counts
- Set-based inverted index

## Cost Analysis

### AWS EC2 Benchmarking Costs

| Instance Type | vCPU | Memory | Cost/hr | Max Dataset |
|---------------|------|--------|---------|-------------|
| t3.large | 2 | 8 GB | $0.08 | 125K docs |
| t3.xlarge | 4 | 16 GB | $0.16 | 291K docs (full) |
| m6i.xlarge | 4 | 16 GB | $0.13 | 291K docs (full) |

Total benchmarking cost for this project: ~$0.50-1.00
- Multiple runs on t3.large
- Testing different dataset sizes

## Correctness Verification

Before and after optimizations, we verified that results remained identical:

```python
# Saved baseline results to verify optimizations don't change output
baseline = load_baseline("results/baseline_results.json")
optimized = run_benchmark()

for query in queries:
    baseline_results = baseline_index.search(query)
    optimized_results = appendable_index.search(query)
    assert results_identical(baseline_results, optimized_results)
```

All optimizations preserved correctness - search results are identical.

## Conclusion

The optimized `AppendableIndex` achieves:

1. Appendability: Can add documents incrementally (regular Index cannot)
2. Reasonable indexing overhead: 1.3-1.4x slower
3. Superior search performance: 50-76x faster than regular Index
4. Scalability: Performance advantage increases with dataset size

The trade-offs (memory usage, indexing time) are acceptable for use cases that require:
- Incremental document addition
- Fast search performance
- Large document collections

## Running the Benchmarks

### Local (small dataset)

```bash
cd benchmark
uv run python run_benchmark.py
```

### AWS (full dataset)

```bash
# 1. Launch EC2 instance (t3.xlarge recommended for full dataset)
# 2. SSH into instance
ssh -i ~/.ssh/your-key.pem ubuntu@<instance-ip>

# 3. Install uv and dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/alexeygrigorev/minsearch.git
cd minsearch/benchmark

# 4. Download and parse Wikipedia
uv run python download_wikipedia.py
uv run python parse_wikipedia.py data/simplewiki-*.xml.bz2

# 5. Run benchmark
uv run python run_full_benchmark.py
```

See `benchmark/AWS.md` for detailed AWS setup instructions.
