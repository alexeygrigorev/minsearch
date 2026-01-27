"""
Highlight utility functions for extracting text snippets around matching terms.
"""

import re


def extract_snippet(text, query_tokens, fragment_size=150, number_of_fragments=1, pre_tag="", post_tag=""):
    """
    Extract snippet(s) from text around matching query tokens.
    
    Args:
        text (str): The text to extract snippets from.
        query_tokens (list): List of tokens to search for in the text.
        fragment_size (int): Maximum size of each fragment in characters. Defaults to 150.
        number_of_fragments (int): Number of fragments to return. Defaults to 1.
        pre_tag (str): Tag to insert before highlighted terms. Defaults to empty string.
        post_tag (str): Tag to insert after highlighted terms. Defaults to empty string.
    
    Returns:
        str: The extracted snippet(s), with matching terms surrounded by pre_tag and post_tag.
              Multiple fragments are joined by " ... ".
    """
    if not text or not query_tokens:
        return ""
    
    text_lower = text.lower()
    
    # Find all positions of query tokens in the text
    token_positions = []
    for token in query_tokens:
        token_lower = token.lower()
        # Find all occurrences of the token
        start = 0
        while True:
            # Use word boundary matching to find whole word occurrences
            match = re.search(r'\b' + re.escape(token_lower) + r'\b', text_lower[start:])
            if not match:
                break
            pos = start + match.start()
            token_positions.append((pos, pos + len(token_lower)))
            start = pos + 1
    
    if not token_positions:
        # No matches found, return beginning of text
        return text[:fragment_size] + ("..." if len(text) > fragment_size else "")
    
    # Sort positions by occurrence
    token_positions.sort()
    
    # Group nearby positions into fragments
    fragments = []
    used_positions = set()
    
    for pos, end_pos in token_positions:
        if (pos, end_pos) in used_positions:
            continue
        
        # Calculate fragment boundaries
        # Try to center the fragment around the match
        half_size = fragment_size // 2
        fragment_start = max(0, pos - half_size)
        fragment_end = min(len(text), fragment_start + fragment_size)
        
        # Adjust start if we're at the end of the text
        if fragment_end - fragment_start < fragment_size:
            fragment_start = max(0, fragment_end - fragment_size)
        
        # Try to break at word boundaries
        if fragment_start > 0:
            # Look for a space after the start position
            space_pos = text.find(' ', fragment_start, min(fragment_start + 20, fragment_end))
            if space_pos != -1:
                fragment_start = space_pos + 1
        
        if fragment_end < len(text):
            # Look for a space before the end position
            space_pos = text.rfind(' ', max(fragment_start, fragment_end - 20), fragment_end)
            if space_pos != -1:
                fragment_end = space_pos
        
        # Extract fragment
        fragment_text = text[fragment_start:fragment_end]
        
        # Highlight matching tokens within this fragment
        if pre_tag or post_tag:
            for token in query_tokens:
                # Use word boundary matching and case-insensitive replacement
                pattern = r'\b(' + re.escape(token) + r')\b'
                fragment_text = re.sub(
                    pattern,
                    lambda m: pre_tag + m.group(1) + post_tag,
                    fragment_text,
                    flags=re.IGNORECASE
                )
        
        # Mark positions as used
        for p, e in token_positions:
            if fragment_start <= p < fragment_end:
                used_positions.add((p, e))
        
        fragments.append({
            'start': fragment_start,
            'text': fragment_text,
            'priority': pos  # Use position for sorting
        })
        
        if len(fragments) >= number_of_fragments:
            break
    
    # Sort fragments by their position in the original text
    fragments.sort(key=lambda f: f['start'])
    
    # Combine fragments
    result_fragments = []
    for i, frag in enumerate(fragments):
        text = frag['text']
        # Add ellipsis if not at the beginning/end of the document
        if frag['start'] > 0:
            text = "..." + text
        if frag['start'] + len(frag['text']) < len(text):
            text = text + "..."
        result_fragments.append(text)
    
    return " ... ".join(result_fragments) if result_fragments else ""


def apply_highlight(doc, text_fields, query_tokens, highlight_config):
    """
    Apply highlighting to a document, returning a new document with snippets instead of full text.
    
    Args:
        doc (dict): The document to highlight.
        text_fields (list): List of text field names to extract snippets from.
        query_tokens (list): List of tokens to search for.
        highlight_config (dict): Configuration for highlighting with fields:
            - fragment_size (int): Maximum size of each fragment. Defaults to 150.
            - number_of_fragments (int): Number of fragments to return. Defaults to 1.
            - pre_tag (str): Tag to insert before highlighted terms. Defaults to empty string.
            - post_tag (str): Tag to insert after highlighted terms. Defaults to empty string.
    
    Returns:
        dict: A new document with snippets for text fields and original values for other fields.
    """
    fragment_size = highlight_config.get('fragment_size', 150)
    number_of_fragments = highlight_config.get('number_of_fragments', 1)
    pre_tag = highlight_config.get('pre_tag', '')
    post_tag = highlight_config.get('post_tag', '')
    
    result = {}
    
    for key, value in doc.items():
        if key in text_fields and isinstance(value, str):
            # Extract snippet for text fields
            snippet = extract_snippet(
                value,
                query_tokens,
                fragment_size=fragment_size,
                number_of_fragments=number_of_fragments,
                pre_tag=pre_tag,
                post_tag=post_tag
            )
            result[key] = snippet
        else:
            # Keep original value for non-text fields
            result[key] = value
    
    return result
