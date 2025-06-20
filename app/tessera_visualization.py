"""tessera_visualization.py
Visualizes NLP results using bar charts and word clouds, and performs multi-task NLP.
"""

from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline

def visualize_word_frequency(text: str) -> None:
    """
    Plots a bar chart of word frequencies in the input text.

    Args:
        text (str): The text to analyze.
    """
    word_counts = Counter(text.split())
    plt.bar(word_counts.keys(), word_counts.values(), color="skyblue")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title("Word Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("word_frequency.png")
    plt.show()

def generate_wordcloud(text: str) -> None:
    """
    Generates and displays a word cloud from the input text.

    Args:
        text (str): The text to visualize.
    """
    wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud")
    plt.savefig("word_cloud.png")
    plt.show()

def multi_task_nlp(text: str) -> dict:
    """
    Performs sentiment analysis, summarization, and named entity recognition on input text.

    Args:
        text (str): The input text.

    Returns:
        dict: NLP results including sentiment, highlighted summary, and named entities.
    """
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    summarization_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

    sentiment = sentiment_pipeline(text)[0]["label"]
    summary = summarization_pipeline(text, max_length=30, min_length=10, do_sample=False)[0]["summary_text"]
    entities = ner_pipeline(summary)

    # Highlight entities in summary
    for entity in entities:
        summary = summary.replace(entity["word"], f"[{entity['word']}]({entity['entity_group']})")

    results = {
        "Sentiment": sentiment,
        "Summary": summary,
        "Named Entities": [{"Entity": e["entity_group"], "Word": e["word"], "Score": e["score"]} for e in entities]
    }
    return results

if __name__ == "__main__":
    text = "Microsoft launched a new AI in 2025, and it’s already changing the tech landscape."
    results = multi_task_nlp(text)

    print("Results:")
    print(f"Sentiment: {results['Sentiment']}")
    print(f"Highlighted Summary: {results['Summary']}")
    print("Named Entities:")
    for entity in results["Named Entities"]:
        print(f"{entity['Entity']}: {entity['Word']} (Score: {entity['Score']:.2f})")

    print("Visualizing Word Frequency:")
    visualize_word_frequency(results["Summary"])

    print("Generating Word Cloud:")
    generate_wordcloud(results["Summary"])