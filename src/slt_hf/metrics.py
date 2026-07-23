from sacrebleu import corpus_bleu


def compute_metrics(eval_preds):
    predictions, labels = eval_preds

    if hasattr(predictions, "tolist"):
        predictions = predictions.tolist()

    if hasattr(labels, "tolist"):
        labels = labels.tolist()

    predictions = [
        " ".join(map(str, pred))
        for pred in predictions
    ]

    references = [
        [" ".join(map(str, label))]
        for label in labels
    ]

    bleu = corpus_bleu(predictions, references)

    return {
        "bleu": bleu.score
    }