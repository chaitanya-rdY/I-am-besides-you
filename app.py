import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.title("AI Agent")

MODEL_ID = "meta-llama/Llama-2-7b-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

prompt = st.text_area("Enter prompt:", height=100)
max_length = st.slider("Max length", 50, 512, 256)

if st.button("Generate"):
    if prompt.strip() != "":
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7, top_p=0.9)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write(response)
