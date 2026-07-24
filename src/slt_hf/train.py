import yaml

from signjoey.batch import Batch
from signjoey.data import load_data, make_data_iter

from .config import SLTConfig
from .model import SLTModel
from .tokenizer import SLTTokenizer
from .train_args import load_training_args
from .trainer import SLTTrainer


class HFPreBatchedDataset:
    """
    SignJoey iterator batch'lerini Hugging Face model girdilerine çevirir.

    Burada her dataset öğesi zaten tam bir batch'tir.
    Bu nedenle Trainer tarafında batch size 1 kullanılacaktır.
    """

    def __init__(
        self,
        dataset,
        txt_pad_index,
        sgn_dim,
        batch_size,
        batch_type="sentence",
        train=False,
        shuffle=False,
    ):
        iterator = make_data_iter(
            dataset=dataset,
            batch_size=batch_size,
            batch_type=batch_type,
            train=train,
            shuffle=shuffle,
        )

        self.batches = []

        for torch_batch in iterator:
            batch = Batch(
                torch_batch=torch_batch,
                txt_pad_index=txt_pad_index,
                sgn_dim=sgn_dim,
                is_train=train,
                use_cuda=False,
            )

            labels = batch.txt.clone()
            labels[labels == txt_pad_index] = -100

            model_inputs = {
                "input_features": batch.sgn,
                "attention_mask": batch.sgn_mask.squeeze(1),
                "decoder_input_ids": batch.txt_input,
                "decoder_attention_mask": batch.txt_mask.squeeze(1),
                "labels": labels,
                "input_lengths": batch.sgn_lengths,
            }

            if batch.gls is not None:
                model_inputs["gloss_labels"] = batch.gls
                model_inputs["gloss_lengths"] = batch.gls_lengths

            self.batches.append(model_inputs)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return self.batches[index]


def prebatched_data_collator(features):
    """
    Dataset'ten gelen her öğe zaten tam bir batch'tir.
    Hugging Face'in tekrar batch boyutu eklemesini engeller.
    """
    if len(features) != 1:
        raise ValueError(
            "HFPreBatchedDataset requires "
            "per_device_train_batch_size=1 and "
            "per_device_eval_batch_size=1."
        )

    return features[0]
def main():

    # -------------------------
    # Load YAML configuration
    # -------------------------
    with open("configs/slt_debug.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # -------------------------
    # Load datasets and vocabularies
    # -------------------------
    train_data, dev_data, test_data, gls_vocab, txt_vocab = load_data(
        cfg["data"]
    )

    # -------------------------
    # Hugging Face tokenizer
    # -------------------------
    tokenizer = SLTTokenizer(
        vocab=dict(txt_vocab.stoi),
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )

    # -------------------------
    # Model configuration
    # -------------------------
    config = SLTConfig(
        feature_size=cfg["data"]["feature_size"],
        hidden_size=cfg["model"]["hidden_size"],
        num_attention_heads=cfg["model"]["num_attention_heads"],
        num_encoder_layers=cfg["model"]["num_encoder_layers"],
        num_decoder_layers=cfg["model"]["num_decoder_layers"],
        intermediate_size=cfg["model"]["intermediate_size"],
        dropout=cfg["model"]["dropout"],
        vocab_size=len(txt_vocab),
        gloss_vocab_size=len(gls_vocab),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
    )

    model = SLTModel(config)
    output_dir = cfg["training"].get(
        "output_dir",
        "checkpoints/slt_debug",
    )

    training_args = load_training_args(
        cfg,
        output_dir=output_dir,
    )

    train_dataset = HFPreBatchedDataset(
        dataset=train_data,
        txt_pad_index=tokenizer.pad_token_id,
        sgn_dim=cfg["data"]["feature_size"],
        batch_size=cfg["training"]["per_device_train_batch_size"],
        batch_type="sentence",
        train=True,
        shuffle=True,
    )

    eval_dataset = HFPreBatchedDataset(
        dataset=dev_data,
        txt_pad_index=tokenizer.pad_token_id,
        sgn_dim=cfg["data"]["feature_size"],
        batch_size=cfg["training"]["per_device_eval_batch_size"],
        batch_type="sentence",
        train=False,
        shuffle=False,
    )

    training_args.per_device_train_batch_size = 1
    training_args.per_device_eval_batch_size = 1

    trainer = SLTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=prebatched_data_collator,
        tokenizer=tokenizer,
        label_smoothing=cfg["training"].get(
            "label_smoothing",
            0.0,
        ),
    )

    trainer.train()

    trainer.save_model(output_dir)

    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()