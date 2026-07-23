from transformers import HfArgumentParser

from .config import SLTConfig
from .model import SLTModel
from .tokenizer import SLTTokenizer
from .trainer import SLTTrainer
from .train_args import (
    ModelArguments,
    DataArguments,
    SLTTrainingArguments,
)


def main():
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            SLTTrainingArguments,
        )
    )

    model_args, data_args, training_args = (
        parser.parse_args_into_dataclasses()
    )

    config = SLTConfig(
        feature_size=model_args.feature_size,
        hidden_size=model_args.hidden_size,
        num_attention_heads=model_args.num_attention_heads,
        num_encoder_layers=model_args.num_encoder_layers,
        num_decoder_layers=model_args.num_decoder_layers,
        intermediate_size=model_args.intermediate_size,
        dropout=model_args.dropout,
        max_position_embeddings=model_args.max_position_embeddings,
        vocab_size=100,
        gloss_vocab_size=50,
    )

    tokenizer = SLTTokenizer()

    model = SLTModel(config)

    trainer = SLTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
    )

    print("SLT HuggingFace pipeline initialized successfully.")


if __name__ == "__main__":
    main()