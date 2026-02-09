# AppendableIndex Performance Improvement Plan

## Current Bottlenecks (1000 docs benchmark)

| Operation | Regular Index | AppendableIndex | Ratio |
|-----------|--------------|-----------------|-------|
| **Indexing** | 3.32s | 47.40s | **14.27x slower** |
| **Search (avg)** | 32.88ms | 882.43ms | **26.84x slower** |

## Root Cause Analysis

### Search Performance Issues (27x slower)

1. **`_create_document_vectors()` (line 341-358)** - Biggest bottleneck
   - Creates document vectors ON-THE-FLY during search
   - Re-tokenizes EVERY matching document for each query
   - `_process_text()` called repeatedly for same docs

2. **`_calculate_tfidf()` (line 290-304)**
   - Recalculates TF-IDF for each (token, doc) pair during search
   - `doc_tokens.count(token)` is O(n) for each token

3. **`_update_inverted_index()` (line 277-288)**
   - `len(set(self.inverted_index[field][token]))` for EVERY token addition
   - Converts list to set just to count unique elements

4. **`_create_query_vector()` (line 314-339)**
   - `query_tokens.count(token)` is O(n) for each token

### Indexing Performance Issues (14x slower)

1. Manual tokenization vs sklearn's optimized C implementation
2. No vectorization in TF-IDF calculation
3. Inverted index using lists with duplicates

## Improvement Plan

### Phase 1: Quick Wins (Low hanging fruit)

#### 1.1 Cache Tokenized Documents
**Impact**: High | **Effort**: Low

```python
# Store tokenized documents during fit/append
self.doc_tokens = []  # List of tokenized documents

# In fit():
for doc in docs:
    tokens = self._process_text(doc.get(field, ""))
    self.doc_tokens.append(tokens)
```

**Benefit**: No re-tokenization during search (~5-10x speedup)

#### 1.2 Use Sets for Inverted Index
**Impact**: Medium | **Effort**: Low

```python
# Instead of:
self.inverted_index[field][token].append(doc_id)
self.doc_frequencies[field][token] = len(set(self.inverted_index[field][token]))

# Use:
self.inverted_index[field][token].add(doc_id)
self.doc_frequencies[field][token] += 1
```

**Benefit**: O(1) insert instead of O(n) set conversion

#### 1.3 Pre-compute IDF Values
**Impact**: Medium | **Effort**: Low

```python
# After fit(), compute IDF for all tokens once
self.idf = {field: {} for field in self.text_fields}
for field in self.text_fields:
    for token, df in self.doc_frequencies[field].items():
        self.idf[field][token] = math.log((self.total_docs + 1) / (df + 1)) + 1
```

### Phase 2: Major Optimizations

#### 2.1 Pre-compute Document Vectors
**Impact**: Very High | **Effort**: Medium

Store normalized TF-IDF vectors for each document during indexing:

```python
# During fit/append, compute and store:
self.doc_vectors = {field: {} for field in self.text_fields}
self.doc_norms = {field: {} for field in self.text_fields}

for doc_id, tokens in enumerate(doc_tokens):
    vector = compute_tfidf_vector(tokens)
    norm = np.linalg.norm(vector)
    self.doc_vectors[field][doc_id] = vector / norm if norm > 0 else vector
```

**Benefit**: Search becomes just cosine similarity on pre-computed vectors (~20x speedup)

#### 2.2 Batch Vectorized Scoring
**Impact**: High | **Effort**: Medium

Instead of computing similarity one document at a time:

```python
# Current: Loop over docs
for doc_id, doc_vector in doc_vectors.items():
    field_scores[doc_id] = np.dot(query_vector, doc_vector)

# Optimized: Matrix multiplication
all_vectors = np.array([self.doc_vectors[field][doc_id] for doc_id in matching_docs])
similarities = all_vectors @ query_vector
```

### Phase 3: Advanced Optimizations

#### 3.1 Use sklearn's HashingVectorizer
**Impact**: High | **Effort**: Medium

- No vocabulary to store
- Stateless - can transform new documents
- Fixed memory footprint
- Very fast

#### 3.2 Sparse Matrix Storage
**Impact**: Medium | **Effort**: High

Store document vectors as scipy sparse matrices instead of dictionaries.

## Expected Performance After Optimizations

| Phase | Indexing Speedup | Search Speedup |
|-------|------------------|----------------|
| Phase 1 | 2-3x | 5-10x |
| Phase 2 | 3-5x | 15-25x |
| Phase 3 | 5-10x | 20-30x |

**Goal**: Get AppendableIndex within 2-3x of regular Index performance.

## Implementation Order

1. ✅ Phase 1.1: Cache tokenized documents
2. ✅ Phase 1.2: Use sets for inverted index
3. ✅ Phase 1.3: Pre-compute IDF values
4. ⬜ Phase 2.1: Pre-compute document vectors
5. ⬜ Phase 2.2: Batch vectorized scoring
6. ⬜ Benchmark after each phase
7. ⬜ Phase 3: Consider sklearn integration if needed
