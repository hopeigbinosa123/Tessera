"""tessera_ner.py
Identifies named entities in text using a transformer-based NER model.
"""

from transformers import pipeline
from typing import List, Dict, Any

def recognize_entities(text: str) -> List[Dict[str, Any]]:
    """
    Extracts named entities from the input text using a pre-trained NER pipeline.

    Args:
        text (str): The text to analyze.

    Returns:
        List[Dict[str, Any]]: A list of entities with labels, words, and confidence scores.
    """
    ner_pipeline = pipeline("ner", grouped_entities=True)
    return ner_pipeline(text)

if __name__ == "__main__":
    sample_text = "Microsoft launched a new AI in 2025 and it’s already popular worldwide."
    entities = recognize_entities(sample_text)

    print("Named Entities:")
    for entity in entities:
        print(f"{entity['entity_group']}: {entity['word']} (Score: {entity['score']:.2f})")