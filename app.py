import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import time
import pandas as pd

st.set_page_config(
    page_title="Sentimind - AI Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .big-font { font-size:20px !important; }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e6e9ef;
        padding: 10px;
        border-radius: 10px;
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

#Model Loading
@st.cache_resource
def load_model():
    model_path = "Shanuka12/sentiment-bert-model"
    device = torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    return tokenizer, model, device

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=100)
    st.title("🧠 Sentimind AI")
    st.markdown("---")
    st.write("This application uses **Google's BERT Model** to analyze emotions in text.")
    st.caption("Created by Shanuka | Powered by Hugging Face")

# Main Page
st.title("📊 AI Product Review Analyzer")
st.markdown("<p class='big-font'>Analyze customer feedback instantly using Artificial Intelligence.</p>", unsafe_allow_html=True)
st.divider()

# Load Model
with st.spinner('🔄 Loading AI Brain...'):
    tokenizer, model, device = load_model()

# --- Prediction Function ---
def make_prediction(text):
    if str(text).strip() == "":
        return "Unknown", 0.0

    encoded_review = tokenizer.encode_plus(
        str(text),
        max_length=128,
        add_special_tokens=True,
        return_token_type_ids=False,
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
    confidence = probs.max().item()
    
    label = "Positive" if prediction.item() == 1 else "Negative"
    return label, confidence

# Coloring Table
def highlight_sentiment(val):
    color = ''
    if val == 'Positive':
        color = 'background-color: #d4edda; color: green; font-weight: bold;'
    elif val == 'Negative':
        color = 'background-color: #f8d7da; color: red; font-weight: bold;'
    return color


if 'analyzed_df' not in st.session_state:
    st.session_state.analyzed_df = None


tab1, tab2 = st.tabs(["✍️ Single Review", "📂 Batch Analysis (CSV)"])

# TAB 1: Single Review Analysis

with tab1:
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("Type your review")
        user_input = st.text_area("", height=200, placeholder="Paste a review here...")
        analyze_button = st.button("🚀 Analyze Sentiment", type="primary")

    with col2:
        st.subheader("Analysis Result")
        if analyze_button and user_input.strip() != "":
            progress_text = "Analyzing..."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.005)
                my_bar.progress(percent_complete + 1, text=progress_text)
            my_bar.empty()

            label, confidence = make_prediction(user_input)
            
            if label == "Positive":
                st.success("## ✅ Positive Review")
                st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
                st.progress(confidence, text="Positive")
                if confidence > 0.9: st.balloons()
            else:
                st.error("## ❌ Negative Review")
                st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
                st.progress(confidence, text="Negative")

        elif analyze_button:
            st.warning("⚠️ Please enter some text!")


# TAB 2: Batch Analysis (CSV Upload) 
with tab2:
    st.subheader("Upload a CSV file")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Column selection
        all_columns = df.columns.tolist()
        text_column = st.selectbox("Select the Review Text Column:", all_columns)
        
        if st.button("🚀 Analyze All Rows"):
            
            # Progress Bar 
            progress_text = "Starting analysis..."
            progress_bar = st.progress(0, text=progress_text)
            
            results = []
            confidences = []
            total_rows = len(df)
            
            for i, row in df.iterrows():
                text = row[text_column]
                label, conf = make_prediction(text)
                results.append(label)
                confidences.append(conf)
                
                current_progress = (i + 1) / total_rows
                progress_message = f"⏳ Processing review {i+1} of {total_rows}..."
                progress_bar.progress(current_progress, text=progress_message)
            
            df['Sentiment'] = results
            df['Confidence Score'] = confidences
            
            st.session_state.analyzed_df = df
            
            progress_bar.empty()
            st.success(f"✅ Analysis Completed! Processed {total_rows} reviews.")

        # Dashboard
        if st.session_state.analyzed_df is not None:
            
            result_df = st.session_state.analyzed_df
            
            st.divider()
            
            # DASHBOARD METRICS
            st.write("### 📊 Analysis Dashboard")
            
            total = len(result_df)
            pos_count = len(result_df[result_df['Sentiment'] == 'Positive'])
            neg_count = len(result_df[result_df['Sentiment'] == 'Negative'])
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Reviews", total, "📝")
            m2.metric("Positive Reviews", pos_count, "✅", delta_color="normal")
            m3.metric("Negative Reviews", neg_count, "❌", delta_color="inverse")
            
            st.divider()

            # CHARTS & FILTERS
            c1, c2 = st.columns([2, 1])
            
            with c1:
                st.write("#### Detailed Results")
                
                filter_option = st.radio("Show:", ["All", "Positive Only", "Negative Only"], horizontal=True)
                
                if filter_option == "Positive Only":
                    filtered_df = result_df[result_df['Sentiment'] == 'Positive']
                elif filter_option == "Negative Only":
                    filtered_df = result_df[result_df['Sentiment'] == 'Negative']
                else:
                    filtered_df = result_df
                
                styled_df = filtered_df.style.map(highlight_sentiment, subset=['Sentiment'])
                
                st.dataframe(
                    styled_df,
                    column_config={
                        "Confidence Score": st.column_config.ProgressColumn(
                            "Confidence",
                            help="How sure the AI is?",
                            format="%.2f",
                            min_value=0,
                            max_value=1,
                        ),
                    },
                    use_container_width=True
                )

            with c2:
                st.write("#### Sentiment Distribution")
                st.bar_chart(result_df['Sentiment'].value_counts())
                
                st.write("#### Download Data")
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name='analyzed_reviews.csv',
                    mime='text/csv',
                    use_container_width=True
                )