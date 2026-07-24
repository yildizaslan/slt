import inspect
from pathlib import Path
from typing import Any, Dict

from transformers import Seq2SeqTrainingArguments


def load_training_args(
    cfg: Dict[str, Any],
    output_dir: str,
) -> Seq2SeqTrainingArguments:
    training_cfg = dict(cfg.get("training", {}))

    valid_parameters = set(
        inspect.signature(
            Seq2SeqTrainingArguments.__init__
        ).parameters
    )

    hf_arguments = {
        key: value
        for key, value in training_cfg.items()
        if key in valid_parameters
    }

    hf_arguments["output_dir"] = str(Path(output_dir))

    return Seq2SeqTrainingArguments(
        **hf_arguments,
    )