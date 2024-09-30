# minsearch

Minimalistic text search engine that uses sklearn and pandas.

This is a simple search library implemented using `sklearn` and `pandas`.

It allows you to index documents with text and keyword fields and perform search queries with support for filtering and boosting.

## Installation 

```bash
pip install minsearch
```


## Environment setup

To run it locally, make sure you have the required dependencies installed:

```bash
pip install pandas scikit-learn
```

Alternatively, use pipenv:

```bash
pipenv install --dev
```

## Usage

Here's how you can use the library:


### Define Your Documents

Prepare your documents as a list of dictionaries. Each dictionary should have the text and keyword fields you want to index.

```python
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
```

### Create the Index

Create an instance of the `Index` class, specifying the text and keyword fields.


```python
from minsearch import Index

index = Index(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)
```

Fit the index with your documents

```python
index.fit(docs)
```

### Perform a Search

Search the index with a query string, optional filter dictionary, and optional boost dictionary.

```python
query = "Can I join the course if it has already started?"

filter_dict = {"course": "data-engineering-zoomcamp"}
boost_dict = {"question": 3, "text": 1, "section": 1}

results = index.search(query, filter_dict, boost_dict)

for result in results:
    print(result)
```

## Notebook 

Run it in a notebook to test it yourself

```bash
pipenv run jupyter notebook
```

## File structure

There's `minsearch` folder and `minsearch.py` file in the root. 

The file `minsearch.py` is kept there because it was used in
the LLM Zoomcamp course, where we'd use `wget` to donwload it.
To avoid breaking changes, we keep the file.



## Publishing

Use `twine` for publishing and `build` for building

```bash
pipenv install --dev twine build
```

Generate a wheel:

```python
pipenv run python -m build
```

Check the packages:

```bash
pipenv run twine check dist/*
```

Upload the library to test PyPI to verify everything is working:

```bash
pipenv run twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

Upload to PyPI:

```bash
pipenv run twine upload dist/*
```

Clean:

```bash
rm -r build/ dist/ minsearch.egg-info/
```

Done!