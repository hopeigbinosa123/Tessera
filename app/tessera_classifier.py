"""tessera_classifier.py
Trains a Naive Bayes classifier for text sentiment categorization.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from typing import List, Tuple

# Example training data
texts = ["I love programming!", "I hate bugs.", "AI is amazing!", "This is so frustrating."]
labels = ["positive", "negative", "positive", "negative"]

def train_classifier(texts: List[str], labels: List[str]) -> Tuple[MultinomialNB, CountVectorizer]:
    """
    Train a Naive Bayes text classifier.

    Args:
        texts (List[str]): Input text samples.
        labels (List[str]): Corresponding text labels.

    Returns:
        Tuple containing the trained model and the CountVectorizer.
    """
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model, vectorizer

def classify_text(model: MultinomialNB, vectorizer: CountVectorizer, text: str) -> str:
    """
    Predict the sentiment category of new text input.

    Args:
        model (MultinomialNB): Trained Naive Bayes model.
        vectorizer (CountVectorizer): Fitted vectorizer.
        text (str): Text to classify.

    Returns:
        str: Predicted category (e.g., 'positive' or 'negative').
    """
    X_new = vectorizer.transform([text])
    return model.predict(X_new)[0]

if __name__ == "__main__":
    model, vectorizer = train_classifier(texts, labels)
    new_text = "I enjoy learning about Tessera!"
    category = classify_text(model, vectorizer, new_text)
    print(f"Category: {category}")