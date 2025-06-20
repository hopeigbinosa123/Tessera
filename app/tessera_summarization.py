from transformers import pipeline

# Function to summarize text with a pre-trained model
def summarize_text(text):
    summarizer = pipeline("summarization")  # Load the summarization pipeline
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    return summary[0]["summary_text"]

# Example usage
if __name__ == "__main__":
    # Input text to summarize
    long_text = """
    Tessera is a groundbreaking AI programming language designed to simplify natural language processing tasks such as sentiment analysis, text classification, and more. 
    By making NLP accessible, Tessera empowers developers of all skill levels to create innovative solutions. 
    Its intuitive syntax and flexibility make it a valuable tool in the AI ecosystem.
    """
    
    # Generate summary
    summary = summarize_text(long_text)

    # Output the summary
    print("Summary:")
    print(summary)  