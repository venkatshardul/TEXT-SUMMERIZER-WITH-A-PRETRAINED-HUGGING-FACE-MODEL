This project focuses on fine-tuning the **T5-Small** (Text-to-Text Transfer Transformer) model for the task of dialogue summarization using the **SAMSum dataset**.

## Project Overview
The goal is to take conversational text (dialogues) and generate concise, human-like summaries. T5 is particularly well-suited for this as it treats every NLP task as a text-to-text problem.

## Dataset
- **Name:** SAMSum Corpus
- **Source:** [Hugging Face / SAMSum](https://huggingface.co/datasets/samsum)
- **Content:** Approximately 16k messenger-like dialogues with accompanying summaries.

## Implementation Steps
1.  **Preprocessing:** 
    - Data cleaning using Regex to handle messenger-specific artifacts (newlines, carriage returns).
    - Normalizing whitespace to improve tokenizer efficiency.
2.  **Tokenization:** 
    - Using `T5Tokenizer` with a max input length of 512 and max target length of 128.
    - Adding the `summarize: ` prefix to all input sequences as required by T5.
3.  **Training:**
    - **Framework:** Hugging Face `Trainer` API.
    - **Hardware:** Trained using NVIDIA Tesla T4 GPU (Google Colab).
    - **Hyperparameters:** 
        - Epochs: 6
        - Learning Rate: 1e-4
        - Batch Size: 8
        - Weight Decay: 0.01
4.  **Evaluation:** Validation loss monitored per epoch.

## How to Use
To generate a summary for a new dialogue:

```python
# Load the saved model
model = T5ForConditionalGeneration.from_pretrained("./saved_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_model")

# Run the inference function
dialogue = "James: Are we still on for the meeting? Sarah: Yes, at 4 PM."
print(summerize_text(dialogue))
```
