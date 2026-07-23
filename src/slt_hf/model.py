import torch
from torch import nn
from transformers import PreTrainedModel

from .config import SLTConfig


class SLTModel(PreTrainedModel):
    """
    HuggingFace-compatible Sign Language Translation model.
    """

    config_class = SLTConfig
    base_model_prefix = "slt"
    main_input_name = "sgn"

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
        sgn,
        txt_input,
        sgn_mask=None,
        txt_mask=None,
        labels=None,
        gloss_labels=None,
        sgn_lengths=None,
        gloss_lengths=None,
    ):
        src = self.input_projection(sgn)
        tgt = self.text_embeddings(txt_input)

        src_key_padding_mask = None
        if sgn_mask is not None:
            src_key_padding_mask = ~sgn_mask.bool()

        tgt_key_padding_mask = None
        if txt_mask is not None:
            tgt_key_padding_mask = ~txt_mask.bool()

        tgt_len = txt_input.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len,
            device=txt_input.device,
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
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=self.config.pad_token_id
            )
            translation_loss = loss_fct(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
            )

        if (
            gloss_labels is not None
            and sgn_lengths is not None
            and gloss_lengths is not None
        ):
            log_probs = gloss_logits.log_softmax(dim=-1).transpose(0, 1)

            ctc_loss_fct = nn.CTCLoss(
                blank=self.config.pad_token_id,
                zero_infinity=True,
            )

            recognition_loss = ctc_loss_fct(
                log_probs,
                gloss_labels,
                sgn_lengths,
                gloss_lengths,
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
