# minsearch

A minimalistic text search engine that uses TF-IDF and cosine similarity for text fields and exact matching for keyword fields. The library provides two implementations:

1. `Index`: A basic search index using scikit-learn's TF-IDF vectorizer
2. `AppendableIndex`: An appendable search index using an inverted index implementation that allows for incremental document addition

## Features

- Text field indexing with TF-IDF and cosine similarity
- Keyword field filtering with exact matching
- Field boosting for fine-tuning search relevance
- Stop word removal and custom tokenization
- Support for incremental document addition (AppendableIndex)
- Customizable tokenizer patterns and stop words
- Efficient search with filtering and boosting

## Installation 

```bash
pip install minsearch
```

## Environment setup

To run it locally, make sure you have the required dependencies installed:

```bash
pip install pandas scikit-learn
```

Alternatively, use uv:

```bash
# Install uv if you haven't already
pip install uv

# Install dependencies
uv sync

# Or install with dev dependencies
uv sync --extra dev
```

## Usage

### Basic Search with Index

```python
from minsearch import Index

# Create documents
docs = [
    {
        "question": "How do I join the course after it has started?",
        "text": "You can join the course at any time. We have recordings available.",
        "section": "General Information",
        "course": "data-engineering-zoomcamp"
    },
    {
        "question": "What are the prerequisites for the course?",
        "text": "You need to have basic knowledge of programming.",
        "section": "Course Requirements",
        "course": "data-engineering-zoomcamp"
    }
]

# Create and fit the index
index = Index(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)
index.fit(docs)

# Search with filters and boosts
query = "Can I join the course if it has already started?"
filter_dict = {"course": "data-engineering-zoomcamp"}
boost_dict = {"question": 3, "text": 1, "section": 1}

results = index.search(query, filter_dict=filter_dict, boost_dict=boost_dict)
```

### Incremental Search with AppendableIndex

```python
from minsearch import AppendableIndex

# Create the index
index = AppendableIndex(
    text_fields=["title", "description"],
    keyword_fields=["course"]
)

# Add documents one by one
doc1 = {"title": "Python Programming", "description": "Learn Python programming", "course": "CS101"}
index.append(doc1)

doc2 = {"title": "Data Science", "description": "Python for data science", "course": "CS102"}
index.append(doc2)

# Search with custom stop words
index = AppendableIndex(
    text_fields=["title", "description"],
    keyword_fields=["course"],
    stop_words={"the", "a", "an"}  # Custom stop words
)
```

### Advanced Features

#### Custom Tokenizer Pattern

```python
from minsearch import AppendableIndex

# Create index with custom tokenizer pattern
index = AppendableIndex(
    text_fields=["title", "description"],
    keyword_fields=["course"],
    tokenizer_pattern=r'[\s\W\d]+'  # Custom pattern to split on whitespace, non-word chars, and digits
)
```

#### Field Boosting

```python
# Boost certain fields to increase their importance in search
boost_dict = {
    "title": 2.0,      # Title matches are twice as important
    "description": 1.0  # Normal importance for description
}
results = index.search("python", boost_dict=boost_dict)
```

#### Keyword Filtering

```python
# Filter results by exact keyword matches
filter_dict = {
    "course": "CS101",
    "level": "beginner"
}
results = index.search("python", filter_dict=filter_dict)
```

## Examples

### Interactive Notebook

The repository includes an interactive Jupyter notebook (`minsearch_example.ipynb`) that demonstrates the library's features using real-world data. The notebook shows:

- Loading and preparing documents from a JSON source
- Creating and configuring the search index
- Performing searches with filters and boosts
- Working with real course-related Q&A data

To run the notebook:

```bash
uv run jupyter notebook
```

Then open `minsearch_example.ipynb` in your browser.

## Development

### Running Tests

```bash
uv run pytest
```

### Building and Publishing

1. Install development dependencies:
```bash
uv sync --extra dev
```

2. Build the package:
```bash
uv run python -m build
```

3. Check the packages:
```bash
uv run twine check dist/*
```

4. Upload to test PyPI:
```bash
uv run twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

5. Upload to PyPI:
```bash
uv run twine upload dist/*
```

6. Clean up:
```bash
rm -r build/ dist/ minsearch.egg-info/
```

## Project Structure

- `minsearch/`: Main package directory
  - `minsearch.py`: Core Index implementation using scikit-learn
  - `append.py`: AppendableIndex implementation with inverted index
- `tests/`: Test suite
- `minsearch_example.ipynb`: Example notebook
- `setup.py`: Package configuration
- `Pipfile`: Development dependencies

Note: The `minsearch.py` file in the root directory is maintained for backward compatibility with the LLM Zoomcamp course.