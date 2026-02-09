#!/usr/bin/env python3
"""
Run full benchmark on Wikipedia dataset.
Compares Regular Index vs AppendableIndex performance.
"""

import sys
import time
import random
import json

sys.path.insert(0, '..')
from minsearch import Index, AppendableIndex


def main():
    print("=" * 70)
    print("FULL WIKIPEDIA BENCHMARK")
    print("=" * 70)

    # Load documents
    print("\nLoading documents from data/wikipedia_docs.jsonl...")
    docs = []
    with open('data/wikipedia_docs.jsonl', 'r') as f:
        for line in f:
            docs.append(json.loads(line))

    print(f"Loaded {len(docs):,} documents")

    total_text_size = sum(len(d.get('text', '')) for d in docs)
    print(f"Total text size: {total_text_size / 1024 / 1024 / 1024:.2f} GB")

    # Benchmark Index
    print("\n" + "-" * 70)
    print("REGULAR INDEX - Indexing")
    print("-" * 70)
    t1 = time.time()
    idx = Index(text_fields=['text'])
    idx.fit(docs)
    index_time = time.time() - t1
    print(f"Time: {index_time:.2f}s")

    # Benchmark AppendableIndex
    print("\n" + "-" * 70)
    print("APPENDABLE INDEX - Indexing")
    print("-" * 70)
    t2 = time.time()
    append_idx = AppendableIndex(text_fields=['text'])
    append_idx.fit(docs)
    append_time = time.time() - t2
    print(f"Time: {append_time:.2f}s")
    print(f"Ratio: {append_time / index_time:.2f}x")

    # Benchmark Search
    print("\n" + "-" * 70)
    print("SEARCH BENCHMARK")
    print("-" * 70)

    # Prepare queries
    random.seed(42)
    titles = [d.get('title', '') for d in docs if d.get('title')]
    queries = [q.split('(')[0].strip().lower() for q in random.sample(titles, 10)]

    # Warmup
    idx.search('test')
    append_idx.search('test')

    # Time searches
    times_index = []
    times_append = []

    for query in queries:
        start = time.time()
        idx.search(query, num_results=10)
        times_index.append(time.time() - start)

        start = time.time()
        append_idx.search(query, num_results=10)
        times_append.append(time.time() - start)

    avg_index = sum(times_index) / len(times_index) * 1000
    avg_append = sum(times_append) / len(times_append) * 1000

    print(f"Queries: {len(queries)}")
    print(f"Regular Index:    {avg_index:.2f}ms avg ({1000/avg_index:.1f} QPS)")
    print(f"AppendableIndex: {avg_append:.2f}ms avg ({1000/avg_append:.1f} QPS)")
    print(f"Ratio: {avg_append / avg_index:.2f}x")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Indexing:  Regular={index_time:.2f}s, Appendable={append_time:.2f}s ({append_time/index_time:.2f}x)")
    print(f"Search:    Regular={avg_index:.2f}ms, Appendable={avg_append:.2f}ms ({avg_append/avg_index:.2f}x)")
    print(f"Indexing is {append_time/index_time:.2f}x slower")
    if avg_append < avg_index:
        print(f"Search is {avg_index/avg_append:.2f}x FASTER!")
    else:
        print(f"Search is {avg_append/avg_index:.2f}x slower")


if __name__ == "__main__":
    main()
