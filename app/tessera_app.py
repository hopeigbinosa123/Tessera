"""tessera_app.py
Streamlit interface for Tessera's NLP capabilities.
"""

import streamlit as st
from transformers import pipeline

# Load NLP pipelines
sentiment_pipeline = pipeline("sentiment-analysis")
summarization_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# --- Streamlit UI ---

st.title("Tessera: AI NLP Companion")

# Text input for multi-task NLP
text = st.text_area("Enter text for NLP tasks:")

if st.button("Analyze"):
    if not text.strip():
        st.error("Please enter text before clicking Analyze.")
    else:
        # Sentiment Analysis
        sentiment = sentiment_pipeline(text)[0]["label"]
        st.subheader("Sentiment Analysis")
        st.write(f"**Sentiment:** {sentiment}")

        # Summarization
        summary = summarization_pipeline(text, max_length=50, min_length=25, do_sample=False)[0]["summary_text"]
        st.subheader("Text Summarization")
        st.write(f"**Summary:** {summary}")

        # Named Entity Recognition
        entities = ner_pipeline(text)
        st.subheader("Named Entity Recognition")
        if entities:
            for entity in entities:
                st.write(f"- **{entity['entity_group']}:** {entity['word']} (Confidence: {entity['score']:.2f})")
        else:
            st.info("No named entities found.")

# --- Question Answering Section ---

st.markdown("---")
st.subheader("Ask a Question")
context = st.text_area("Enter context for Question Answering (e.g., 'Microsoft launched AI in 2025.')")
question = st.text_input("Enter your question:")

if st.button("Ask"):
    if not context.strip():
        st.error("Context cannot be empty. Please provide background text.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        result = qa_pipeline(question=question, context=context)
        st.write("**Answer:**", result["answer"])
        st.caption(f"Confidence: {result['score']:.2f}")