import os
import time
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

def summarize_dataset(dataset, rate_limit=4):
    summaries = []
    count = 0

    for article in dataset['test']['article']:
        if count >= rate_limit:
            time.sleep(60)
            count = 0

        summary = summarize_text(article)
        summaries.append(summary)
        count += 1

    return summaries

if __name__ == "__main__":
    summaries = summarize_dataset(dataset, rate_limit=4)
    for i, summary in enumerate(summaries[:5]):
        print(f"Summary {i+1}:\n{summary}\n")
    print("Summarization completed.")
