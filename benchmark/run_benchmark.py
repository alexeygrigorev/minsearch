#!/usr/bin/env python3
"""
Benchmark script comparing Index vs AppendableIndex performance.
Measures indexing time, search time, and memory usage.
"""

import json
import time
import tracemalloc
from pathlib import Path
from typing import List, Dict, Any
import sys
import numpy as np

# Add parent directory to path to import minsearch
sys.path.insert(0, str(Path(__file__).parent.parent))

from minsearch import Index, AppendableIndex


def measure_time(func, *args, **kwargs):
    """Measure execution time of a function."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start


def measure_memory(func, *args, **kwargs):
    """Measure memory usage of a function."""
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak


def format_time(seconds):
    """Format time in human-readable format."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f} μs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds:.2f} s"


def format_memory(bytes_val):
    """Format memory in human-readable format."""
    mb = bytes_val / (1024 * 1024)
    if mb < 1:
        return f"{bytes_val / 1024:.2f} KB"
    return f"{mb:.2f} MB"


def create_index_regular(docs: List[Dict], text_fields: List[str], keyword_fields: List[str] = None):
    """Create and fit a regular Index."""
    index = Index(
        text_fields=text_fields,
        keyword_fields=keyword_fields or [],
        vectorizer_params={"dtype": np.float32}
    )
    index.fit(docs)
    return index


def create_index_appendable(docs: List[Dict], text_fields: List[str], keyword_fields: List[str] = None):
    """Create and fit an AppendableIndex."""
    index = AppendableIndex(
        text_fields=text_fields,
        keyword_fields=keyword_fields or []
    )
    index.fit(docs)
    return index


def create_index_appendable_incremental(docs: List[Dict], text_fields: List[str], keyword_fields: List[str] = None):
    """Create an AppendableIndex using incremental append."""
    index = AppendableIndex(
        text_fields=text_fields,
        keyword_fields=keyword_fields or []
    )
    # Append documents one by one
    for doc in docs:
        index.append(doc)
    return index


def benchmark_indexing(docs: List[Dict], text_fields: List[str], keyword_fields: List[str] = None):
    """Benchmark indexing performance."""
    print("\n" + "=" * 70)
    print("INDEXING BENCHMARK")
    print("=" * 70)
    print(f"Documents: {len(docs):,}")
    print(f"Text fields: {text_fields}")
    print(f"Keyword fields: {keyword_fields or []}")

    # Calculate total text size
    total_chars = sum(len(str(doc.get(field, ""))) for doc in docs for field in text_fields)
    print(f"Total text size: {format_memory(total_chars)}")

    print("\n" + "-" * 70)

    # Benchmark regular Index
    print("\n1. Regular Index (TfidfVectorizer)")
    _, regular_time = measure_time(create_index_regular, docs, text_fields, keyword_fields)
    regular_index, regular_mem = measure_memory(create_index_regular, docs, text_fields, keyword_fields)

    print(f"   Time:  {format_time(regular_time)}")
    print(f"   Memory: {format_memory(regular_mem)}")

    # Benchmark AppendableIndex (batch fit)
    print("\n2. AppendableIndex (batch fit)")
    _, appendable_time = measure_time(create_index_appendable, docs, text_fields, keyword_fields)
    appendable_index, appendable_mem = measure_memory(create_index_appendable, docs, text_fields, keyword_fields)

    print(f"   Time:  {format_time(appendable_time)}")
    print(f"   Memory: {format_memory(appendable_mem)}")
    print(f"   Ratio vs Regular: {appendable_time / regular_time:.2f}x slower")
    print(f"   Memory ratio: {appendable_mem / regular_mem:.2f}x")

    # Benchmark AppendableIndex (incremental append) - only on smaller datasets
    if len(docs) <= 100000:
        print("\n3. AppendableIndex (incremental append)")
        _, incremental_time = measure_time(create_index_appendable_incremental, docs, text_fields, keyword_fields)
        incremental_index, incremental_mem = measure_memory(create_index_appendable_incremental, docs, text_fields, keyword_fields)

        print(f"   Time:  {format_time(incremental_time)}")
        print(f"   Memory: {format_memory(incremental_mem)}")
        print(f"   Ratio vs batch: {incremental_time / appendable_time:.2f}x slower")

    return {
        'regular_index': regular_index,
        'appendable_index': appendable_index,
        'times': {
            'regular': regular_time,
            'appendable_batch': appendable_time,
            'appendable_incremental': incremental_time if len(docs) <= 100000 else None
        },
        'memory': {
            'regular': regular_mem,
            'appendable_batch': appendable_mem,
            'appendable_incremental': incremental_mem if len(docs) <= 100000 else None
        }
    }


def benchmark_search(indexes: Dict, queries: List[str]):
    """Benchmark search performance."""
    print("\n" + "=" * 70)
    print("SEARCH BENCHMARK")
    print("=" * 70)
    print(f"Queries: {len(queries)}")

    results = {}

    for index_name, index in indexes.items():
        print(f"\n{index_name}:")
        index_results = []

        # Warm up
        index.search(queries[0])

        times = []
        for query in queries:
            _, search_time = measure_time(index.search, query, num_results=10)
            times.append(search_time)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"   Average: {format_time(avg_time)}")
        print(f"   Min:     {format_time(min_time)}")
        print(f"   Max:     {format_time(max_time)}")
        print(f"   QPS:     {1/avg_time:.2f} queries/second")

        results[index_name] = {
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'qps': 1/avg_time
        }

    return results


def run_comprehensive_benchmark(docs: List[Dict], num_search_queries: int = 100):
    """Run a comprehensive benchmark comparing Index vs AppendableIndex."""

    text_fields = ['text']
    keyword_fields = None

    print("\n" + "=" * 70)
    print("MINSEARCH BENCHMARK - Index vs AppendableIndex")
    print("=" * 70)

    # Run indexing benchmark
    indexing_results = benchmark_indexing(docs, text_fields, keyword_fields)

    # Generate search queries (sample from document titles)
    print("\n" + "=" * 70)
    print("GENERATING SEARCH QUERIES")
    print("=" * 70)

    titles = [doc['title'] for doc in docs if 'title' in doc]
    import random
    random.seed(42)
    sample_titles = random.sample(titles, min(num_search_queries, len(titles)))
    queries = [title.split('(')[0].strip().lower() for title in sample_titles]

    print(f"Generated {len(queries)} search queries from document titles")

    # Run search benchmark
    search_results = benchmark_search({
        'Regular Index': indexing_results['regular_index'],
        'AppendableIndex (batch)': indexing_results['appendable_index'],
    }, queries)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nIndexing:")
    print(f"  Regular Index:           {format_time(indexing_results['times']['regular'])}")
    print(f"  AppendableIndex (batch): {format_time(indexing_results['times']['appendable_batch'])}")

    print("\nMemory:")
    print(f"  Regular Index:           {format_memory(indexing_results['memory']['regular'])}")
    print(f"  AppendableIndex (batch): {format_memory(indexing_results['memory']['appendable_batch'])}")

    print("\nSearch (average):")
    for name, stats in search_results.items():
        print(f"  {name}: {format_time(stats['avg_time'])} ({stats['qps']:.2f} QPS)")

    return {
        'indexing': indexing_results['times'],
        'memory': indexing_results['memory'],
        'search': {k: v['avg_time'] for k, v in search_results.items()},
        'qps': {k: v['qps'] for k, v in search_results.items()}
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark minsearch Index vs AppendableIndex')
    parser.add_argument('-i', '--input', default='data/wikipedia_docs.jsonl',
                        help='Path to JSON/JSONL file with documents')
    parser.add_argument('-n', '--num-docs', type=int, default=None,
                        help='Maximum number of documents to use (default: all)')
    parser.add_argument('-q', '--num-queries', type=int, default=100,
                        help='Number of search queries to benchmark (default: 100)')

    args = parser.parse_args()

    # Load documents
    input_path = Path(args.input)
    if not input_path.is_absolute():
        script_dir = Path(__file__).parent
        input_path = script_dir / args.input

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        print("\nPlease run:")
        print("  1. python download_wikipedia.py")
        print("  2. python parse_wikipedia.py <downloaded-file>")
        sys.exit(1)

    print(f"Loading documents from: {input_path}")

    # Determine if JSON or JSONL
    if str(input_path).endswith('.jsonl'):
        # Load from JSONL (streaming)
        docs = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
                if args.num_docs and len(docs) >= args.num_docs:
                    break
    else:
        # Load from JSON
        with open(input_path, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        if args.num_docs:
            docs = docs[:args.num_docs]

    print(f"Loaded {len(docs):,} documents")

    # Run benchmark
    results = run_comprehensive_benchmark(docs, args.num_queries)

    # Save results
    results_path = Path(__file__).parent / "data" / "benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
