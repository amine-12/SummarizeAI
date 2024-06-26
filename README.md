# SummarizeAI

SummarizeAI is an AI-powered service that automatically generates concise summaries from long-form text using the Cohere API. This project includes scripts for data collection, preprocessing, training, and summarization.

## Overview

This project aims to develop an automated text summarization service that:
- Accepts long-form text (e.g., articles, reports).
- Generates concise summaries.
- Uses Cohere's NLP capabilities for summarization.

## Project Structure

- `data/`: Scripts for data collection.
  - `download_datasets.py`: Script to download the datasets.
- `preprocessing/`: Scripts for data preprocessing.
  - `preprocess_data.py`: Script to preprocess the datasets.
- `training/`: Scripts for training and evaluating the model.
- `cohere_summarize.py`: Script to summarize text using Cohere's API.
- `requirements.txt`: Required Python libraries.
- `README.md`: Project documentation.

## Setup

### Prerequisites

- Python 3.7 or higher
- Cohere API key 

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/amine-12/SummarizeAI.git
   cd SummarizeAI
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On macOS/Linux
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your Cohere API key:
   - Create a `.env` file in the root directory and add your API key:
     ```env
     COHERE_API_KEY=your_api_key_here
     ```

### Data Collection

Download the datasets:
```bash
python data/download_datasets.py
```

### Data Preprocessing

Preprocess the data:
```bash
python preprocessing/preprocess_data.py
```

### Summarization

Run the summarization:
```bash
python cohere_summarize.py
```

### Example Output

The summarization script will print the first 5 summaries generated from the dataset. Here is an example output:

```
Summary 1:
[Generated summary]

Summary 2:
[Generated summary]

Summary 3:
[Generated summary]

Summary 4:
[Generated summary]

Summary 5:
[Generated summary]

Summarization completed.
```

