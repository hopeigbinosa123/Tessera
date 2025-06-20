"""tessera_qa.py
Performs extractive question answering using a transformer model.
"""

from transformers import pipeline

def question_answering(context: str, question: str) -> dict:
    """
    Answers a question based on a given context using a pre-trained transformer.

    Args:
        context (str): Background information containing the answer.
        question (str): The question to be answered.

    Returns:
        dict: A dictionary with answer text and confidence score.
    """
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    result = qa_pipeline(question=question, context=context)

    if result["score"] > 0.7:
        return {"Answer": result["answer"], "Confidence": result["score"]}
    return {"Answer": "No confident answer found.", "Confidence": result["score"]}

if __name__ == "__main__":
    context = input("Enter context (text with information): ")
    question = input("Enter your question: ")
    result = question_answering(context, question)

    print("\nResults:")
    print(f"Question: {question}")
    print(f"Answer: {result['Answer']}")
    print(f"Confidence: {result['Confidence']:.2f}")