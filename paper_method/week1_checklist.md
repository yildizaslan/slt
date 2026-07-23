# Week 1 — Paper to Code Checklist

Paper:
Camgoz et al. (2020)
Sign Language Transformers:
Joint End-to-end Sign Language Recognition and Translation

---

## 1. Input Representation

Paper: The paper takes a sign-language video sequence as input. Each video is represented as a sequence of frame-level visual features. These features are projected through a spatial embedding layer before being passed to the Transformer encoder.

Repository: The repository does not train directly from raw video pixels inside the Transformer. It loads pre-extracted visual feature sequences from the dataset pipeline. The input tensor has the general form [batch_size, sequence_length, feature_dimension], together with sequence lengths and masks.

Notes:The raw data source is video, but the Transformer receives pre-computed visual features rather than RGB frames. Glosses are not the input; they are supervision targets for the recognition branch.

---

## 2. Encoder

Paper:
The paper uses a Transformer encoder to model the temporal relationships between sign-language frame features. Positional encoding is added to preserve sequence order.

Repository:
The encoder is implemented as a Transformer encoder in the SignJoey framework. It receives embedded visual features and produces contextualized feature representations for both recognition and translation.

Notes:
The encoder is shared by both tasks (Sign Language Recognition and Sign Language Translation).

---

## 3. Decoder

Paper:
The translation decoder is a Transformer decoder. It attends to the encoder outputs and previously generated target tokens to produce the translated spoken-language sentence.

Repository:
The repository implements an autoregressive Transformer decoder for translation. During training, teacher forcing is used by feeding the ground-truth target sequence into the decoder.

Notes:
Unlike the recognition branch, the decoder generates natural language text token by token.

---

## 4. Recognition Head

Paper:
The recognition branch predicts gloss sequences from the shared encoder outputs. It is trained using Connectionist Temporal Classification (CTC) loss.

Repository:
The repository implements a gloss prediction head on top of the encoder. The predicted gloss logits are passed to the CTC loss during training.

Notes:
The recognition branch does not require frame-level alignment because CTC learns the alignment automatically.

---

## 5. Translation Head

Paper:
The translation branch generates spoken-language sentences from the shared encoder representations using a Transformer decoder. The recognition and translation tasks are trained jointly.

Repository:
The repository uses the decoder outputs to predict the target sentence vocabulary through a linear output layer followed by softmax.

Notes:
The translation head benefits from the shared encoder, allowing recognition and translation to improve each other during joint training.

---

## 6. Loss Functions

Paper:
The model is trained jointly using two objectives: CTC loss for sign language recognition and Cross-Entropy loss for sign language translation. The total training loss is a weighted combination of both losses.

Repository:
The training script computes both recognition and translation losses during each training step and combines them according to the configuration.

Notes:
The multi-task objective allows the shared encoder to learn representations that are useful for both recognition and translation.

---

## 7. Training Pipeline

Paper:
The model is trained end-to-end using mini-batch optimization. During training, the encoder is shared between recognition and translation, and both objectives are optimized jointly.

Repository:
The training process is managed by the SignJoey training pipeline. Configuration files define hyperparameters such as learning rate, batch size, optimizer, scheduler, and number of epochs.

Notes:
The repository separates configuration from implementation, making experiments reproducible and easy to modify.:

---

## 8. Inference

Paper:
During inference, the encoder first processes the input sign-language feature sequence. The translation decoder then generates the spoken-language sentence autoregressively until an end-of-sequence token is produced.

Repository:
The repository performs beam search decoding to generate the final translation. The decoder predicts one token at a time while attending to the encoder outputs.

Notes:
Beam search improves translation quality compared to greedy decoding by exploring multiple candidate sequences.

---

## 9. Questions / Ambiguities

-## 9. Questions / Ambiguities

- Some implementation details are not explained in the paper and can only be found in the repository.

- The paper does not explain in detail how the recognition and translation loss weights were selected.

- The repository includes implementation and compatibility changes that are not described in the paper.

