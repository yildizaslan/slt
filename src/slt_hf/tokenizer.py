import json
import os

from transformers import PreTrainedTokenizer


class SLTTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.json"}

    def __init__(self, vocab_file=None, vocab=None, **kwargs):
        if vocab_file is not None:
            with open(vocab_file, "r", encoding="utf-8") as file:
                vocab = json.load(file)

        if vocab is None:
            vocab = {
                "<pad>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "<unk>": 3,
            }

        self.vocab = vocab
        self.ids_to_tokens = {
            token_id: token for token, token_id in vocab.items()
        }

        super().__init__(
            pad_token=kwargs.pop("pad_token", "<pad>"),
            bos_token=kwargs.pop("bos_token", "<bos>"),
            eos_token=kwargs.pop("eos_token", "<eos>"),
            unk_token=kwargs.pop("unk_token", "<unk>"),
            **kwargs,
        )

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def _tokenize(self, text):
        return text.split()

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0,
        token_ids_1=None,
    ):
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

        if token_ids_1 is not None:
            output += token_ids_1 + [self.eos_token_id]

        return output

    def save_vocabulary(self, save_directory, filename_prefix=None):
        os.makedirs(save_directory, exist_ok=True)

        filename = "vocab.json"
        if filename_prefix:
            filename = f"{filename_prefix}-{filename}"

        vocab_path = os.path.join(save_directory, filename)

        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(
                self.vocab,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return (vocab_path,)