"""tessera_bert.py
Performs sentiment analysis using a pre-trained BERT model.
"""

from transformers import pipeline

def analyze_sentiment_with_bert(text: str) -> str:
    """
    Analyze sentiment of input text using a BERT-based classifier.

    Args:
        text (str): Input string to evaluate.

    Returns:
        str: Sentiment label ('POSITIVE' or 'NEGATIVE').
    """
    classifier = pipeline("sentiment-analysis")
    result = classifier(text)
    return result[0]["label"]

if __name__ == "__main__":
    example = "Tessera is the best thing since sliced bread!"
    sentiment = analyze_sentiment_with_bert(example)
    print(f"Sentiment: {sentiment}")