# AI Review Analyzer 🤖

This is a Sentiment Analysis web application built using **Streamlit** and **Hugging Face Transformers**. It uses a fine-tuned BERT model to classify text reviews as either **Positive** or **Negative**.

## 🚀 Features

- **Real-time Analysis**: Instantly analyzes the sentiment of user input.
- **User-Friendly Interface**: Simple web interface built with Streamlit.
- **Pre-trained Model**: Utilizes a locally saved BERT model for accurate predictions.

## 🛠️ Tech Stack

- **Python**
- **Streamlit** (Web Framework)
- **PyTorch** (Deep Learning Framework)
- **Transformers** (Hugging Face Library)

## � Dataset

The model is trained on the **Amazon Reviews for Sentiment Analysis** dataset.
You can download it using `kagglehub`:

```python
path = kagglehub.dataset_download("bittlingmayer/amazonreviews")
```

## �📂 Project Structure

```
Product-Sentiment/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── model/                  # Directory containing the fine-tuned model files
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── vocab.txt
└── notebook/
    └── advance_sentiment.ipynb  # Notebook used for model training/analysis
```

## ⚙️ Installation & Setup

1. **Clone the repository** (if applicable) or navigate to the project folder:

   ```bash
   cd Product-Sentiment
   ```

2. **Create a Virtual Environment** (Recommended):

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ How to Run

Run the Streamlit application using the following command:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

## 🧠 Model Information

The application loads a pre-trained BERT model from the `./model` directory. Ensure that all model files (`config.json`, `model.safetensors`, `vocab.txt`, etc.) are present in this folder for the app to function correctly.

## 📝 License

This project is open-source and available for educational purposes.
