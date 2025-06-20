"""tessera_sentiment.py
Performs sentiment analysis using TextBlob.
"""

from textblob import TextBlob

def analyze_sentiment(text: str) -> str:
    """
    Analyze the sentiment of the input text using TextBlob.

    Args:
        text (str): The input string.

    Returns:
        str: One of 'Positive', 'Negative', or 'Neutral'.
    """
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    return "Neutral"

if __name__ == "__main__":
    text = "I absolutely love Tessera—it’s so intuitive!"
    sentiment = analyze_sentiment(text)
    print(f"Sentiment: {sentiment}")