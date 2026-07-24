import torch.nn.functional as F
from transformers import Seq2SeqTrainer


class SLTTrainer(Seq2SeqTrainer):
    """
    HuggingFace Trainer for SLT with custom label smoothing.
    """

    def __init__(
        self,
        *args,
        label_smoothing=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.slt_label_smoothing = float(label_smoothing)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs,
    ):
        outputs = model(**inputs)

        labels = inputs.get("labels")
        recognition_loss = outputs.get("recognition_loss")

        if (
            labels is not None
            and self.slt_label_smoothing > 0.0
        ):
            logits = outputs["logits"]

            translation_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                label_smoothing=self.slt_label_smoothing,
            )

            if recognition_loss is not None:
                loss = translation_loss + recognition_loss
            else:
                loss = translation_loss
        else:
            loss = outputs["loss"]

        if loss is None:
            raise ValueError(
                "Model did not return a loss. "
                "Make sure labels are included in the batch."
            )

        if return_outputs:
            return loss, outputs

        return loss