#!/usr/bin/env python3
"""
Save baseline results from Index and AppendableIndex for correctness validation.
Later we can compare optimized results to ensure correctness is maintained.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path to import minsearch
sys.path.insert(0, str(Path(__file__).parent.parent))

from minsearch import Index, AppendableIndex


def save_results(docs, queries, output_dir):
    """Save results from both indices for comparison."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'num_docs': len(docs),
        'queries': queries,
        'index': {},
        'appendable': {}
    }

    # Test Regular Index
    print("Creating Regular Index...")
    idx = Index(text_fields=['text', 'title'], keyword_fields=[])
    idx.fit(docs)

    print("Running queries on Regular Index...")
    for query in queries:
        results['index'][query] = idx.search(query, num_results=10)

    # Test AppendableIndex
    print("Creating AppendableIndex...")
    append_idx = AppendableIndex(text_fields=['text', 'title'], keyword_fields=[])
    append_idx.fit(docs)

    print("Running queries on AppendableIndex...")
    for query in queries:
        results['appendable'][query] = append_idx.search(query, num_results=10)

    # Save results
    output_path = output_dir / 'baseline_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved baseline results to: {output_path}")

    # Compare results
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON (Index vs AppendableIndex)")
    print("=" * 70)

    for query in queries:
        index_results = results['index'][query]
        appendable_results = results['appendable'][query]

        # Get top document IDs/titles
        index_titles = [r.get('title', r.get('text', '')[:50]) for r in index_results[:5]]
        appendable_titles = [r.get('title', r.get('text', '')[:50]) for r in appendable_results[:5]]

        print(f"\nQuery: '{query}'")
        print(f"  Index top 5: {index_titles}")
        print(f"  Appendable top 5: {appendable_titles}")

        # Check if results are similar
        if index_titles == appendable_titles:
            print("  [OK] Results are identical")
        else:
            print("  [INFO] Results differ (expected - different implementations)")

    return results


def main():
    # Load documents
    data_path = Path(__file__).parent / 'data' / 'wikipedia_docs_1000.jsonl'

    if not data_path.exists():
        print(f"Error: {data_path} not found")
        print("Please create the test data first:")
        print("  head -n 1000 benchmark/data/wikipedia_docs.jsonl > benchmark/data/wikipedia_docs_1000.jsonl")
        sys.exit(1)

    print(f"Loading documents from: {data_path}")
    docs = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    print(f"Loaded {len(docs)} documents")

    # Define test queries
    queries = [
        'python programming language',
        'machine learning',
        'world history',
        'climate change',
        'space exploration'
    ]

    # Save baseline results
    save_results(docs, queries, Path(__file__).parent / 'results')


if __name__ == "__main__":
    main()
