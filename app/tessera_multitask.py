"""tessera_multitask.py
Performs sentiment analysis, summarization, and named entity recognition in one function.
"""

from transformers import pipeline
from typing import Dict, List, Any

def multi_task_nlp(text: str) -> Dict[str, Any]:
    """
    Perform multiple NLP tasks (sentiment analysis, summarization, NER) on input text.

    Args:
        text (str): The input text to process.

    Returns:
        dict: Results of sentiment, summary, and named entities.
    """
    sentiment_pipeline = pipeline("sentiment-analysis")
    summarization_pipeline = pipeline("summarization")
    ner_pipeline = pipeline("ner", grouped_entities=True)

    sentiment = sentiment_pipeline(text)[0]["label"]
    summary_result = summarization_pipeline(
        text,
        max_length=50,
        min_length=25,
        do_sample=False
    )
    summary = summary_result[0]["summary_text"]
    entities = ner_pipeline(text)

    results = {
        "Sentiment": sentiment,
        "Summary": summary,
        "Named Entities": [
            {
                "Entity": entity["entity_group"],
                "Word": entity["word"],
                "Score": entity["score"]
            }
            for entity in entities
        ]
    }
    return results

if __name__ == "__main__":
    sample_text = "Microsoft launched a new AI in 2025, and it’s already changing the tech landscape."
    output = multi_task_nlp(sample_text)

    print("Results:")
    print(f"Sentiment: {output['Sentiment']}")
    print(f"Summary: {output['Summary']}")
    print("Named Entities:")
    for entity in output["Named Entities"]:
        print(f"{entity['Entity']}: {entity['Word']} (Score: {entity['Score']:.2f})")