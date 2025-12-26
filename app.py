import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🤖",
    layout="centered"
)

@st.cache_resource
def load_model():
    model_path = "Shanuka12/sentiment-bert-model"
    
    device = torch.device('cpu')
    
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    
    return tokenizer, model, device

with st.spinner('Loading AI Model... Please wait...'):
    tokenizer, model, device = load_model()

st.title("🤖 AI Review Analyzer")
st.markdown("Type a review below to see if it's **Positive** or **Negative**!")

user_input = st.text_area("Enter your text here:", height=150, placeholder="Example: This product is amazing!")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        encoded_review = tokenizer.encode_plus(
            user_input,
            max_length=128,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True, 
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        input_ids = encoded_review['input_ids'].to(device)
        attention_mask = encoded_review['attention_mask'].to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)

        probs = F.softmax(output.logits, dim=1)
        _, prediction = torch.max(probs, dim=1)
        
        confidence = probs.max().item() * 100
        
        st.markdown("---")
        if prediction.item() == 1:
            st.success(f"### Result: Positive 😃")
            st.write(f"Confidence: **{confidence:.2f}%**")
        else:
            st.error(f"### Result: Negative 😞")
            st.write(f"Confidence: **{confidence:.2f}%**")