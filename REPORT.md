# Sessions 7–8 Report

In Sessions 7 and 8, the Sign Language Translation project was migrated to the Hugging Face framework. This included implementing the Hugging Face configuration, model, tokenizer, trainer, training arguments, and evaluation components. The training pipeline was also updated to work with the new Hugging Face structure while keeping the original project design.

Most of the required implementation was completed successfully. During development, several issues related to configuration values, package dependencies, and training settings were identified and resolved. These steps helped build a better understanding of how the different parts of the project work together.

The main challenge appeared during the final training stage. Although the implementation was completed, the original Phoenix2014T feature files could no longer be downloaded because the provided links were unavailable. As a result, the dataset contained empty sign feature tensors, and the training process could not continue. After checking the dataset, the data loader, and the model, it was confirmed that the problem was caused by the missing dataset files rather than by the implementation itself.

For this assignment, AI was used as a learning and tutoring tool. It was mainly used to explain concepts, understand the Hugging Face framework, debug errors, and better understand the connection between the research paper and the implementation. The implementation itself was completed step by step while using these explanations to better understand the code and the overall architecture.