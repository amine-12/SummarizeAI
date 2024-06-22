import os
import cohere
from datasets import load_from_disk
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv('COHERE_API_KEY')

# Initialize Cohere client
co = cohere.Client(api_key)

# Load the dataset
dataset = load_from_disk('data/cnn_dailymail')  # Load the original dataset

def summarize_text(text):
    response = co.summarize(
        text=text,
        length='medium',  # You can change to 'short', 'long', or 'auto'
        format='paragraph',  # You can change to 'bullets' or 'auto'
        model='command',  # You can change to 'command-nightly', 'command-light', etc.
        extractiveness='low',  # You can change to 'medium', 'high', or 'auto'
        temperature=0.3
    )
    summary = response.summary  # Correct attribute access
    return summary

def summarize_dataset(dataset):
    summaries = []
    for article in dataset['test']['article']:
        summary = summarize_text(article)
        summaries.append(summary)
    return summaries

if __name__ == "__main__":
    summaries = summarize_dataset(dataset)
    for i, summary in enumerate(summaries[:5]):  # Print the first 5 summaries
        print(f"Summary {i+1}:\n{summary}\n")
    print("Summarization completed.")
