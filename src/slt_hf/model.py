import torch
from torch import nn
from transformers import PreTrainedModel

from .config import SLTConfig


class SLTModel(PreTrainedModel):
    """
    HuggingFace-compatible Sign Language Translation model.

    Standard HF names:
        input_features
        attention_mask
        decoder_input_ids
        decoder_attention_mask
        labels

    Legacy SignJoey-style aliases are also supported:
        sgn
        sgn_mask
        txt_input
        txt_mask
    """

    config_class = SLTConfig
    base_model_prefix = "slt"
    main_input_name = "input_features"

    def __init__(self, config: SLTConfig):
        super().__init__(config)

        self.input_projection = nn.Linear(
            config.feature_size,
            config.hidden_size,
        )

        self.text_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
        )

        self.transformer = nn.Transformer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            batch_first=True,
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        self.gloss_head = nn.Linear(
            config.hidden_size,
            config.gloss_vocab_size,
        )

        self.post_init()

    def forward(
        self,
        input_features=None,
        decoder_input_ids=None,
        attention_mask=None,
        decoder_attention_mask=None,
        labels=None,
        gloss_labels=None,
        input_lengths=None,
        gloss_lengths=None,
        # Backward-compatible aliases
        sgn=None,
        txt_input=None,
        sgn_mask=None,
        txt_mask=None,
        sgn_lengths=None,
        **kwargs,
    ):
        """
        Shapes:
            input_features       : (B, S, feature_size)
            attention_mask       : (B, S)
            decoder_input_ids    : (B, T)
            decoder_attention_mask: (B, T)
            labels               : (B, T)
            logits               : (B, T, vocab_size)
            gloss_logits         : (B, S, gloss_vocab_size)
        """

        # Preserve compatibility with the previous SignJoey-style API.
        if input_features is None:
            input_features = sgn

        if decoder_input_ids is None:
            decoder_input_ids = txt_input

        if attention_mask is None:
            attention_mask = sgn_mask

        if decoder_attention_mask is None:
            decoder_attention_mask = txt_mask

        if input_lengths is None:
            input_lengths = sgn_lengths

        if input_features is None:
            raise ValueError(
                "input_features must be provided."
            )

        if decoder_input_ids is None:
            raise ValueError(
                "decoder_input_ids must be provided."
            )

        # (B, S, feature_size) -> (B, S, hidden_size)
        src = self.input_projection(input_features)

        # (B, T) -> (B, T, hidden_size)
        tgt = self.text_embeddings(decoder_input_ids)

        src_key_padding_mask = None
        if attention_mask is not None:
            # HF mask: 1 means valid, 0 means padding.
            # PyTorch padding mask: True means ignore.
            src_key_padding_mask = ~attention_mask.bool()

        tgt_key_padding_mask = None
        if decoder_attention_mask is not None:
            tgt_key_padding_mask = ~decoder_attention_mask.bool()

        tgt_len = decoder_input_ids.size(1)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len,
            device=decoder_input_ids.device,
        )

        memory = self.transformer.encoder(
            src,
            src_key_padding_mask=src_key_padding_mask,
        )

        decoder_output = self.transformer.decoder(
            tgt,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        logits = self.lm_head(decoder_output)
        gloss_logits = self.gloss_head(memory)

        loss = None
        translation_loss = None
        recognition_loss = None

        if labels is not None:
            translation_loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )

        if (
            gloss_labels is not None
            and input_lengths is not None
            and gloss_lengths is not None
        ):
            # CTCLoss expects (S, B, C).
            log_probs = gloss_logits.log_softmax(dim=-1).transpose(0, 1)

            recognition_loss = nn.functional.ctc_loss(
                log_probs,
                gloss_labels,
                input_lengths,
                gloss_lengths,
                blank=self.config.pad_token_id,
                zero_infinity=True,
            )

        if translation_loss is not None and recognition_loss is not None:
            loss = translation_loss + recognition_loss
        elif translation_loss is not None:
            loss = translation_loss
        elif recognition_loss is not None:
            loss = recognition_loss

        return {
            "loss": loss,
            "logits": logits,
            "gloss_logits": gloss_logits,
            "translation_loss": translation_loss,
            "recognition_loss": recognition_loss,
        }