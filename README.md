# 🧠 AI Agent for Student Request Classification  

This project fine-tunes **LLaMA-2 (7B)** using **QLoRA** and **PEFT** to classify student requests into one of **41 predefined categories** (e.g., *Bonafide for Passport, Course Withdrawal, Transcript Request*).  

---

## 📌 Architecture  

### Components  
- **User Interface (UI):** Built using **Streamlit** for easy interaction.  
- **Model Backend:** Fine-tuned **LLaMA-2 (7B)** using **QLoRA**.  
- **Tokenizer:** Hugging Face `AutoTokenizer`.  
- **Fine-Tuning Framework:** **PEFT (LoRA adapters)** for parameter-efficient fine-tuning.  
- **Evaluation Module:** Uses **scikit-learn** (`accuracy_score`, `f1_score`).  
- **Persistence Layer:** Model + LoRA adapters stored in `/results` or Hugging Face Hub.  

### Interaction Flow  
1. User enters request text in Streamlit.  
2. Text → Tokenizer → Fine-tuned model.  
3. Model outputs logits → mapped to one of **41 labels**.  
4. Predicted label returned to UI.  

---

## 📊 Data Science Report  

### Fine-Tuning Setup  
- **Dataset:** Custom CSV (`text,label`) with **41 request categories**.  
- **Split:** 80% train / 20% test.  
- **Model:** `meta-llama/Llama-2-7b-hf`.  
- **Training Config:**  
  - QLoRA adapters (`q_proj`, `v_proj`)  
  - Optimizer: AdamW, LR=2e-4  
  - Batch size = 4  
  - Epochs = 3  

---

## ✅ Evaluation  

- **Quantitative:**  
  - `accuracy_score`, `f1_score` (macro avg)  
- **Qualitative (Examples):**  
  - Input: *“I need a Bonafide for Passport application.”* → `Bonafide for Passport` ✅  
  - Input: *“Please allow me to stay on campus in summer.”* → `Permission to stay on Campus during summer` ✅  

---

## 💻 Streamlit App  

Run the app locally:  

```bash
streamlit run app.py
