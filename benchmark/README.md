# Benchmark Results

## Dataset

- **Source**: Simple Wikipedia dump (2026-02-01)
- **Total documents**: 291,737 articles
- **Total size**: 1.0 GB (JSONL format)
- **Full dataset**: `data/wikipedia_docs.jsonl` (not in git)
- **Test subset**: `data/wikipedia_docs_1000.jsonl` (1000 docs, 9.0 MB)

## Benchmark Results (1000 documents)

| Operation | Regular Index (TfidfVectorizer) | AppendableIndex | Ratio |
|-----------|-------------------------------|-----------------|-------|
| **Indexing** | 3.32s | 47.40s | **14.27x slower** |
| **Search (avg)** | 32.88ms | 882.43ms | **26.84x slower** |

## Performance Analysis

The AppendableIndex is significantly slower than the regular Index:

### Indexing Bottlenecks (14x slower)
1. **Manual TF-IDF calculation** vs sklearn's optimized C implementation
2. **Per-document processing** without vectorization
3. **Tokenizer overhead** - custom Python tokenizer vs sklearn's

### Search Bottlenecks (27x slower)
1. **On-the-fly vector creation** - document vectors computed during search instead of pre-computed
2. **Manual cosine similarity** vs sklearn's optimized matrix operations
3. **Nested Python loops** for matching documents and calculating scores

## Usage

### Download Wikipedia dump
```bash
cd benchmark
python download_wikipedia.py
```

### Parse Wikipedia dump to JSONL
```bash
python parse_wikipedia.py data/simplewiki-*.xml.bz2
```

### Run benchmark
```bash
# Full benchmark with custom parameters
python run_benchmark.py -n 1000 -q 20

# Using the small subset
python run_benchmark.py -i data/wikipedia_docs_1000.jsonl
```

### Quick test
```bash
cd benchmark
python -c "
from minsearch import Index, AppendableIndex
import json

docs = [json.loads(line) for line in open('data/wikipedia_docs_1000.jsonl')]

# Regular Index
idx = Index(text_fields=['text'])
idx.fit(docs)

# AppendableIndex
append_idx = AppendableIndex(text_fields=['text'])
append_idx.fit(docs)

# Compare search
results1 = idx.search('python programming')
results2 = append_idx.search('python programming')
"
```

## Improvement Opportunities

1. **Pre-compute document vectors** during indexing instead of on-the-fly during search
2. **Use numpy vectorization** for TF-IDF calculations
3. **Cache query token processing**
4. **Consider using sklearn's TfidfVectorizer** with incremental learning
5. **Optimize the inverted index lookup** with better data structures
