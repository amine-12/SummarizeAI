from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_from_disk, load_metric

def evaluate_model():
    datasets = load_from_disk('data/final_cnn_dailymail')

    model_name = './summarization_model'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    rouge = load_metric('rouge')

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = [[label] for label in labels]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {key: value.mid.fmeasure * 100 for key, value in result.items()}

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    results = trainer.evaluate(eval_dataset=datasets['test'])
    print("Evaluation results:", results)

if __name__ == "__main__":
    evaluate_model()
    print("Model evaluation completed.")
