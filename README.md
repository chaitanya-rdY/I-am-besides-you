Architecture
Components

User Interface (UI): Built using Streamlit for easy interaction.

Model Backend: Fine-tuned LLaMA-2 (7B) using QLoRA.

Tokenizer: Hugging Face AutoTokenizer.

Fine-Tuning Framework: PEFT (LoRA adapters) for parameter-efficient fine-tuning.

Evaluation Module: Uses scikit-learn (accuracy_score, f1_score).

Persistence Layer: Model + LoRA adapters stored in /results or Hugging Face Hub.

Interaction Flow

User enters request text in Streamlit.

Text → Tokenizer → Fine-tuned model.

Model outputs logits → mapped to one of 41 labels.

Predicted label returned to UI.
