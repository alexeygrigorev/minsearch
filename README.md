# minsearch

A minimalistic search engine that provides both text-based and vector-based search capabilities. The library provides three implementations:

1. `Index`: A basic search index using scikit-learn's TF-IDF vectorizer for text fields
2. `AppendableIndex`: An appendable search index using an inverted index implementation that allows for incremental document addition
3. `VectorSearch`: A vector search index using cosine similarity for pre-computed vectors

## Features

- Text field indexing with TF-IDF and cosine similarity
- Vector search with cosine similarity for pre-computed embeddings
- Field boosting for fine-tuning search relevance (text-based search)
- Extensive filtering capabilities (exact match and ranges)
- Support for incremental document addition (AppendableIndex and VectorSearch)
- Customizable tokenizer patterns and stop words
- Result highlighting with configurable formatting

## Installation 

We recommend to use `uv`: 

```bash
uv add minsearch
```

Or, install with `pip`:

```bash
pip install minsearch
```


**Note:** minsearch requires Python 3.10 or later.

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

# Search
results = index.search("python programming")
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

### Incremental Vector Search

`VectorSearch` also supports appending vectors incrementally:

```python
from minsearch import VectorSearch
import numpy as np

# Create the index
index = VectorSearch(keyword_fields=["category", "level"])

# Append a single vector
vector = np.random.rand(768)
doc = {"id": 1, "title": "Python Tutorial", "category": "programming", "level": "beginner"}
index.append(vector, doc)

# Append multiple vectors in batch
vectors = np.random.rand(10, 768)
payload = [
    {"id": i+2, "title": f"Document {i+2}", "category": "data", "level": "intermediate"}
    for i in range(10)
]
index.append_batch(vectors, payload)

# Search works the same way
query_vector = np.random.rand(768)
results = index.search(query_vector, num_results=5)
```


### Custom Tokenizer

```python
from minsearch import AppendableIndex
from minsearch.tokenizer import Tokenizer

tokenizer = Tokenizer(
    stop_words='english',  # Use default English stop words
    stemmer='porter'       # Apply Porter stemming
)

index = AppendableIndex(
    text_fields=["title", "description"],
    keyword_fields=["course"],
    tokenizer=tokenizer
)
```

### Field Boosting (Text-based Search)

```python
# Boost certain fields to increase their importance in search
boost_dict = {
    "title": 2.0,       # Title matches are twice as important
    "description": 1.0  # Normal importance for description
}
results = index.search("python", boost_dict=boost_dict)
```

### Filtering

All filter types (keyword, numeric range, date/time range) work with all search index types: `Index`, `AppendableIndex`, and `VectorSearch`.

#### Keyword Filtering

Filter results by exact keyword matches:

```python
filter_dict = {
    "course": "CS101",
    "level": "beginner"
}
results = index.search("python", filter_dict=filter_dict)
```

#### Numeric Range Filtering

Filter results by numeric values using comparison operators:

```python
from minsearch import AppendableIndex

docs = [
    {"title": "Python Basics", "price": 29.99, "rating": 4.5},
    {"title": "Advanced Python", "price": 49.99, "rating": 4.8},
    {"title": "Python Masterclass", "price": 99.99, "rating": 4.9},
]

index = AppendableIndex(
    text_fields=["title"],
    keyword_fields=[],
    numeric_fields=["price", "rating"]
)
index.fit(docs)

# Price greater than or equal to 40
results = index.search("python", filter_dict={"price": [(">=", 40)]})

# Rating between 4.5 and 4.9
results = index.search("python", filter_dict={"rating": [(">=", 4.5), ("<=", 4.9)]})

# Multiple numeric filters
results = index.search(
    "python",
    filter_dict={
        "price": [("<", 100)],
        "rating": [(">=", 4.7)]
    }
)
```

Supported operators: `==` (equals), `!=` (not equals), `>` (greater than), `>=` (greater than or equal), `<` (less than), `<=` (less than or equal).

#### Date/Time Range Filtering

Filter results by date and time values:

```python
from datetime import datetime, date
from minsearch import AppendableIndex

docs = [
    {
        "title": "Python Course",
        "start_date": date(2024, 1, 15),
        "created_at": datetime(2023, 12, 1, 10, 30)
    },
    {
        "title": "Data Science Course",
        "start_date": date(2024, 2, 1),
        "created_at": datetime(2023, 12, 15, 14, 0)
    },
]

index = AppendableIndex(
    text_fields=["title"],
    keyword_fields=[],
    date_fields=["start_date", "created_at"]
)
index.fit(docs)

# Courses starting after a specific date
results = index.search(
    "python",
    filter_dict={"start_date": [(">", date(2024, 1, 1))]}
)

# Courses created in a date range
results = index.search(
    "course",
    filter_dict={
        "created_at": [
            (">=", datetime(2023, 12, 1)),
            ("<=", datetime(2023, 12, 31))
        ]
    }
)
```

Date fields accept `date`, `datetime`, or `pandas.Timestamp` objects.

#### Combined Filtering

You can combine keyword, numeric, and date filters in a single query:

```python
results = index.search(
    "python course",
    filter_dict={
        "level": "advanced",                      # Keyword filter
        "price": [("<", 100)],                    # Numeric filter
        "start_date": [(">", date(2024, 1, 1))]   # Date filter
    }
)
```

### Result Highlighting

The `Highlighter` class works with search results from any index type (`Index`, `AppendableIndex`, or `VectorSearch`). It extracts highlighted snippets from search results, showing where the query terms match in the text:

```python
from minsearch import AppendableIndex, Highlighter, Tokenizer

# Create documents
docs = [
    {
        "question": "How do I join the course after it has started?",
        "text": "You can join the course at any time. We have recordings available for all sessions.",
        "course": "data-engineering-zoomcamp"
    },
    {
        "question": "Can I get a refund if I drop the course?",
        "text": "Refunds are available within the first 30 days of enrollment.",
        "course": "data-engineering-zoomcamp"
    }
]

# Create and fit the index
tokenizer = Tokenizer(
    stop_words='english',
    stemmer='porter'
)

index = AppendableIndex(
    text_fields=["question", "text"],
    keyword_fields=["course"],
    tokenizer=tokenizer
)
index.fit(docs)

# Search
results = index.search("join course", num_results=1)

# Create highlighter
highlighter = Highlighter(
    highlight_fields=["question", "text"],
    skip_fields=["course"],
    max_matches=3,
    snippet_size=150,
    highlight_format="**",  # Bold with markdown
    tokenizer=tokenizer
)

# Highlight results
highlighted = highlighter.highlight("join course", results)
```

Example output:

```json
{
    "question": {
        "matches": ["How do I **join** the **course** after it has started?"],
        "total_matches": 1
    },
    "text": {
        "matches": ["You can **join** the **course** at any time. We have recordings available for..."],
        "total_matches": 1
    },
    "course": "data-engineering-zoomcamp"
}
```

Highlighter options:

- `highlight_fields`: List of field names to extract highlights from
- `skip_fields`: List of field names to exclude from output (pass-through only)
- `max_matches`: Maximum number of matches to return per field (default: 5)
- `snippet_size`: Maximum characters per match snippet (default: 200)
- `highlight_format`: Format for highlights - can be a string delimiter, tuple (open, close), or callable
- `tokenizer`: Tokenizer to use (must match the index's tokenizer for best results)

Custom highlight formats:

```python
# Markdown bold (default)
highlighter = Highlighter(..., highlight_format="**")  # **text**

# HTML
highlighter = Highlighter(..., highlight_format=("<b>", "</b>"))  # <b>text</b>

# ANSI for terminal
highlighter = Highlighter(..., highlight_format="\033[1m")  # text (bold)

# Custom function
highlighter = Highlighter(..., highlight_format=lambda t: f"[{t}]")  # [text]
```

### Stemming

Stemming reduces words to their root form, improving search recall by matching different word forms. For example, "running", "runs", and "ran" all stem to "run".

```python
from minsearch import AppendableIndex
from minsearch.tokenizer import Tokenizer

# Use stemming with the default English stop words
tokenizer = Tokenizer(
    stop_words='english',
    stemmer='snowball'  # Options: 'porter', 'snowball', 'lancaster', or None
)

index = AppendableIndex(
    text_fields=["title", "description"],
    keyword_fields=["course"],
    tokenizer=tokenizer
)

# Now "joining" will match "join", "joined", "joins", etc.
results = index.search("joining the course")
```

### Stemmer Comparison

- **Porter** - Original Porter algorithm (1980)
  - Well-established, fast, good for English
  - Some edge cases, less aggressive than Snowball
  - Good default for general use

- **Snowball** (Porter2) - Improved Porter algorithm
  - Handles more edge cases, more accurate stemming, has official specification
  - Slightly slower than Porter
  - **Recommended** - best overall accuracy

- **Lancaster** - Very aggressive stemming
  - Reduces words to shortest stems, good for recall
  - Can over-stem, may produce non-words
  - Use when maximizing recall is critical

- **None** - No stemming
  - Preserves original words, fastest
  - No morphological matching
  - Use for exact matching only

**Recommendation**: Use the `snowball` stemmer for best overall accuracy. It's based on the [official Snowball specification](https://snowballstem.org/algorithms/english/stemmer.html) and handles more edge cases than Porter while being less aggressive than Lancaster.

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
