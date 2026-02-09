#!/usr/bin/env python3
"""
Parse Simple Wikipedia XML dump and extract documents for benchmarking.
Extracts page titles and text content, saving as JSON.
"""

import bz2
import json
import xml.sax
from pathlib import Path
from typing import List, Dict
import sys


class StopParsing(Exception):
    """Exception to stop parsing when max_docs is reached."""
    pass


class WikipediaHandler(xml.sax.ContentHandler):
    """SAX handler for parsing Wikipedia XML dump."""

    def __init__(self, output_file, max_docs=None):
        super().__init__()
        self.output_file = output_file
        self.docs = []
        self.max_docs = max_docs
        self.current_page = {}
        self.current_text = []
        self.in_page = False
        self.in_title = False
        self.in_text = False
        self.in_revision = False
        self.skip_pages = 0  # Skip special pages (User:, Talk:, etc.)
        self.doc_count = 0

    def startElement(self, name, attrs):
        if name == "page":
            self.in_page = True
            self.current_page = {}
            self.current_text = []
        elif self.in_page:
            if name == "title":
                self.in_title = True
                self.current_text_title = []
            elif name == "text":
                self.in_text = True
                self.current_text_content = []
            elif name == "revision":
                self.in_revision = True

    def endElement(self, name):
        if name == "page":
            self.in_page = False

            # Skip special pages
            if "title" in self.current_page:
                title = self.current_page["title"]
                if ":" in title and title.split(":")[0] in [
                    "User", "User talk", "Wikipedia", "File",
                    "File talk", "Template", "Template talk",
                    "Help", "Help talk", "Category", "Category talk",
                    "Portal", "Talk", "MediaWiki"
                ]:
                    self.skip_pages += 1
                    return

            # Only add pages with both title and text
            if "title" in self.current_page and "text" in self.current_page:
                text = self.current_page["text"].strip()
                if len(text) > 50:  # Skip very short pages
                    doc = {
                        "title": self.current_page["title"],
                        "text": text,
                        "url": f"https://simple.wikipedia.org/wiki/{self.current_page['title'].replace(' ', '_')}"
                    }
                    # Write immediately to JSONL
                    self.output_file.write(json.dumps(doc) + "\n")
                    self.doc_count += 1

                    # Check if we've reached max_docs
                    if self.max_docs and self.doc_count >= self.max_docs:
                        raise StopParsing()

        elif self.in_page:
            if name == "title":
                self.in_title = False
                self.current_page["title"] = "".join(self.current_text_title)
            elif name == "text":
                self.in_text = False
                self.current_page["text"] = "".join(self.current_text_content)
            elif name == "revision":
                self.in_revision = False

    def characters(self, content):
        if self.in_title:
            self.current_text_title.append(content)
        elif self.in_text:
            self.current_text_content.append(content)


def parse_wikipedia_dump(filepath: str, max_docs: int = None, output_path: str = None):
    """
    Parse Wikipedia XML dump and extract documents to JSONL.

    Args:
        filepath: Path to the Wikipedia dump file (.bz2 or .xml)
        max_docs: Maximum number of documents to extract (None for all)
        output_path: Path to save the parsed documents as JSONL

    Returns:
        Number of documents extracted
    """
    print(f"Parsing: {filepath}")

    # Default output path
    if output_path is None:
        data_dir = Path(__file__).parent / "data"
        output_path = data_dir / "wikipedia_docs.jsonl"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine if file is bz2 compressed
    if filepath.endswith('.bz2'):
        open_func = lambda f: bz2.open(f, 'rt', encoding='utf-8')
    else:
        open_func = lambda f: open(f, 'r', encoding='utf-8')

    doc_count = 0
    skip_count = 0

    with open_func(filepath) as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            handler = WikipediaHandler(f_out, max_docs=max_docs)

            print("Parsing XML (writing to JSONL incrementally)...")

            try:
                xml.sax.parse(f_in, handler)
            except StopParsing:
                print(f"\nReached max_docs limit ({max_docs})")

            doc_count = handler.doc_count
            skip_count = handler.skip_pages

    print(f"Extracted {doc_count} documents")
    print(f"Skipped {skip_count} special pages")
    print(f"Saved to: {output_path}")

    # Calculate file size
    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Output file size: {output_size_mb:.1f} MB")

    return doc_count


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Parse Wikipedia XML dump')
    parser.add_argument('input', help='Path to Wikipedia dump file (.bz2 or .xml)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output JSONL file path (default: data/wikipedia_docs.jsonl)')
    parser.add_argument('-n', '--max-docs', type=int, default=None,
                        help='Maximum number of documents to extract (default: all)')

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        data_dir = Path(__file__).parent / "data"
        args.output = data_dir / "wikipedia_docs.jsonl"

    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        # Try relative to script directory
        script_dir = Path(__file__).parent
        input_path = script_dir / "data" / args.input
        if not input_path.exists():
            print(f"Error: File not found: {args.input}")
            sys.exit(1)

    # Parse the dump
    doc_count = parse_wikipedia_dump(
        str(input_path),
        max_docs=args.max_docs,
        output_path=args.output
    )

    print(f"\nDone! Extracted {doc_count} documents.")


if __name__ == "__main__":
    main()
