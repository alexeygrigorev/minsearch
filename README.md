# minsearch

A minimalistic search engine that provides both text-based and vector-based search capabilities. The library provides three implementations:

1. `Index`: A basic search index using scikit-learn's TF-IDF vectorizer for text fields
2. `AppendableIndex`: An appendable search index using an inverted index implementation that allows for incremental document addition
3. `VectorSearch`: A vector search index using cosine similarity for pre-computed vectors

## Features

- Text field indexing with TF-IDF and cosine similarity
- Vector search with cosine similarity for pre-computed embeddings
- Keyword field filtering with exact matching
- Field boosting for fine-tuning search relevance (text-based search)
- Stop word removal and custom tokenization
- Support for incremental document addition (AppendableIndex)
- Customizable tokenizer patterns and stop words
- Efficient search with filtering and boosting

## Installation 

```bash
pip install minsearch
```

## Environment setup

For development purposes, use uv:

```bash
# Install uv if you haven't already
pip install uv
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

### Vector Search with VectorSearch

```python
from minsearch import VectorSearch
import numpy as np

# Create sample vectors and payload documents
vectors = np.random.rand(100, 768)  # 100 documents, 768-dimensional vectors
payload = [
    {"id": 1, "title": "Python Tutorial", "category": "programming", "level": "beginner"},
    {"id": 2, "title": "Data Science Guide", "category": "data", "level": "intermediate"},
    {"id": 3, "title": "Machine Learning Basics", "category": "ai", "level": "advanced"},
    # ... more documents
]

# Create and fit the vector search index
index = VectorSearch(keyword_fields=["category", "level"])
index.fit(vectors, payload)

# Search with a query vector
query_vector = np.random.rand(768)  # 768-dimensional query vector
filter_dict = {"category": "programming", "level": "beginner"}

results = index.search(query_vector, filter_dict=filter_dict, num_results=5)
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

#### Field Boosting (Text-based Search)

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
uv run hatch build
```

3. Publish to test PyPI:
```bash
uv run hatch publish --repo test
```

4. Publish to PyPI:
```bash
uv run hatch publish
```

5. Clean up:
```bash
rm -r dist/
```

Or run 

```bash
python publish.py
```

Note: For Hatch publishing, you'll need to configure your PyPI credentials in `~/.pypirc` or use environment variables.

## PyPI Credentials Setup

Create a `.pypirc` file in your home directory with your PyPI credentials:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-your-main-api-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-test-api-token-here
```

**Important Notes:**
- Use `__token__` as the username for API tokens
- Get your tokens from [PyPI](https://pypi.org/manage/account/token/) and [Test PyPI](https://test.pypi.org/manage/account/token/)
- Set file permissions: `chmod 600 ~/.pypirc`

**Alternative: Environment Variables**
```bash
export HATCH_INDEX_USER=__token__
export HATCH_INDEX_AUTH=your-pypi-token
```

## Project Structure

- `minsearch/`: Main package directory
  - `minsearch.py`: Core Index implementation using scikit-learn
  - `append.py`: AppendableIndex implementation with inverted index
  - `vector.py`: VectorSearch implementation using cosine similarity
- `tests/`: Test suite
- `minsearch_example.ipynb`: Example notebook
- `setup.py`: Package configuration
- `Pipfile`: Development dependencies

Note: The `minsearch.py` file in the root directory is maintained for backward compatibility with the LLM Zoomcamp course.
