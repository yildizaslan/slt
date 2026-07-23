# Week 1 Notes
## 1. What are the inputs to the model?
The model takes sign language video features as input. The features are extracted from the video frames and then passed to the Transformer encoder.

## 2. What encoder does the paper use? What does the repo use?
The paper uses a Transformer encoder. The repository also uses a Transformer encoder in the SignJoey framework.

## 3. What decoder variant? How does it differ from the vanilla Transformer decoder?
The paper uses a Transformer decoder for translation. The repository also uses a Transformer decoder. The decoder generates the translation while sharing the encoder with the recognition task.

## 4. What is the vocabulary? Glosses or text or both?
The model uses both glosses and text. Glosses are used for sign language recognition, and text is used for sign language translation.

## 5. What is the loss function? Is it the same as cross-entropy?
The model uses two loss functions. It uses CTC loss for sign language recognition and Cross-Entropy loss for translation. The final loss is a combination of both, so it is not only Cross-Entropy.

## 6. What data augmentation or regularisation does the paper use?
The paper uses dropout as regularisation. The repository also supports label smoothing during training.

## 7. What is the evaluation metric and how is it computed?
The model uses BLEU for translation and WER for recognition. BLEU compares the generated sentence with the reference translation. WER measures the errors between the predicted glosses and the reference glosses.

## 8. What is the weakest part of the architecture in your opinion?
The model depends on pre-extracted visual features. Learning directly from video could improve the model.

## 9. What would you try first if you had two more weeks?
I would try a newer Transformer model and compare the results with the original implementation.
