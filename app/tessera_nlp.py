"""tessera_nlp.py
Performs basic NLP preprocessing: stopword removal, stemming, and lemmatization.
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from typing import List

# Download required resources (run once)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

def remove_stopwords(text: str) -> str:
    """
    Removes English stopwords from the input text.

    Args:
        text (str): Input sentence or paragraph.

    Returns:
        str: Text without stopwords.
    """
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

def stem_words(text: str) -> str:
    """
    Applies stemming to words in the input text.

    Args:
        text (str): Preprocessed text.

    Returns:
        str: Stemmed version of the text.
    """
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    stemmed = [stemmer.stem(word) for word in words]
    return " ".join(stemmed)

def lemmatize_words(text: str) -> str:
    """
    Applies lemmatization to words in the input text.

    Args:
        text (str): Preprocessed text.

    Returns:
        str: Lemmatized version of the text.
    """
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized)

if __name__ == "__main__":
    sample_text = "Tessera is an exciting AI programming language for natural language processing."

    no_stopwords = remove_stopwords(sample_text)
    print("After Stopword Removal:")
    print(no_stopwords)

    stemmed = stem_words(no_stopwords)
    print("\nAfter Stemming:")
    print(stemmed)

    lemmatized = lemmatize_words(no_stopwords)
    print("\nAfter Lemmatization:")
    print(lemmatized)