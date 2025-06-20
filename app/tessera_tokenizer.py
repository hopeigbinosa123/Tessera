"""tessera_tokenizer.py
A basic regex-based tokenizer for quick text preprocessing.
"""

import re
from typing import List

def tokenize(text: str) -> List[str]:
    """
    Tokenizes input text using word boundaries.

    Args:
        text (str): Input sentence or paragraph.

    Returns:
        List[str]: List of individual tokens.
    """
    return re.findall(r'\b\w+\b', text)

if __name__ == "__main__":
    sample = "Tessera is an exciting AI programming language!"
    tokens = tokenize(sample)
    print(tokens)