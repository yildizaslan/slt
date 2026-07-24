import evaluate
import numpy as np


def make_compute_metrics(tokenizer):
    bleu_metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = np.asarray(predictions)
        labels = np.asarray(labels)

        labels = np.where(
            labels == -100,
            tokenizer.pad_token_id,
            labels,
        )

        decoded_predictions = tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
        )

        decoded_labels = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
        )

        decoded_predictions = [
            text.strip()
            for text in decoded_predictions
        ]

        decoded_labels = [
            [text.strip()]
            for text in decoded_labels
        ]

        result = bleu_metric.compute(
            predictions=decoded_predictions,
            references=decoded_labels,
        )

        return {
            "bleu": result["score"],
        }

    return compute_metrics