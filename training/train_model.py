
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk

def train_model():
    datasets = load_from_disk('data/final_cnn_dailymail')

    model_name = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained('./summarization_model')
    tokenizer.save_pretrained('./summarization_model')

if __name__ == "__main__":
    train_model()
    print("Model training completed and saved.")
