import os
import cohere
from datasets import load_from_disk
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('COHERE_API_KEY')

co = cohere.Client(api_key)

dataset = load_from_disk('data/cnn_dailymail')

def summarize_text(text):
    response = co.summarize(
        text=text,
        length='medium',
        format='paragraph',
        model='command',
        extractiveness='low',
        temperature=0.3
    )
    summary = response.summary
    return summary

def summarize_dataset(dataset):
    summaries = []
    for article in dataset['test']['article']:
        summary = summarize_text(article)
        summaries.append(summary)
    return summaries

if __name__ == "__main__":
    summaries = summarize_dataset(dataset)
    for i, summary in enumerate(summaries[:5]):
        print(f"Summary {i+1}:\n{summary}\n")
    print("Summarization completed.")
