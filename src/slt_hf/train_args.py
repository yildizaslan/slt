from dataclasses import dataclass, field
from typing import Optional

from transformers import Seq2SeqTrainingArguments


@dataclass
class ModelArguments:
    feature_size: int = field(default=1024)
    hidden_size: int = field(default=512)
    num_attention_heads: int = field(default=8)
    num_encoder_layers: int = field(default=3)
    num_decoder_layers: int = field(default=3)
    intermediate_size: int = field(default=2048)
    dropout: float = field(default=0.1)
    max_position_embeddings: int = field(default=400)


@dataclass
class DataArguments:
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    vocab_file: Optional[str] = field(default=None)
    max_source_length: int = field(default=400)
    max_target_length: int = field(default=100)


@dataclass
class SLTTrainingArguments(Seq2SeqTrainingArguments):
    recognition_loss_weight: float = field(default=1.0)
    translation_loss_weight: float = field(default=1.0)