from transformers import BartTokenizer
from datasets import load_from_disk, DatasetDict


def preprocess_function(examples, tokenizer):
    inputs = tokenizer(examples['article'], max_length=1024, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['highlights'], max_length=128, truncation=True)
    examples['input_ids'] = inputs['input_ids']
    examples['attention_mask'] = inputs['attention_mask']
    examples['labels'] = labels['input_ids']
    return examples


def preprocess_dataset():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    dataset = load_from_disk('data/cnn_dailymail')
    tokenized_datasets = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True,
                                     remove_columns=['article', 'highlights'])
    tokenized_datasets.save_to_disk('data/tokenized_cnn_dailymail')

    train_test_split = tokenized_datasets['train'].train_test_split(test_size=0.2)
    val_test_split = train_test_split['test'].train_test_split(test_size=0.5)

    train_dataset = train_test_split['train']
    val_dataset = val_test_split['train']
    test_dataset = val_test_split['test']

    final_datasets = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    final_datasets.save_to_disk('data/final_cnn_dailymail')


if __name__ == "__main__":
    preprocess_dataset()
    print("Data preprocessing completed and saved to disk.")
