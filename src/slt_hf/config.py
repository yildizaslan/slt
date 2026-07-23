from transformers import PretrainedConfig


class SLTConfig(PretrainedConfig):
    model_type = "slt"

    def __init__(
        self,
        feature_size=1024,
        hidden_size=512,
        num_attention_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        intermediate_size=2048,
        dropout=0.1,
        max_position_embeddings=400,
        vocab_size=0,
        gloss_vocab_size=0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        kwargs.setdefault("decoder_start_token_id", bos_token_id)
        kwargs.setdefault("is_encoder_decoder", True)
        kwargs.setdefault("tie_word_embeddings", False)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.gloss_vocab_size = gloss_vocab_size