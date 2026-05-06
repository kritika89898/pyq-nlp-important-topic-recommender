# PYQ-Based Important Topic Recommendation System using NLP

This is a Streamlit web app for analyzing previous year questions and recommending important topics using NLP.

## Features
- Upload CSV/Excel PYQ dataset
- Load Kaggle GATE CSE Question Classification Dataset using `kagglehub`
- Text preprocessing
- BERT/Sentence Transformer embeddings
- TF-IDF fallback
- Similar question detection
- Topic frequency analysis
- Topic modeling using LDA
- Syllabus mapping using semantic similarity
- Topic classification using TF-IDF + Logistic Regression
- Important topic recommendation
- CSV report export

## Dataset format
Recommended columns:

```text
question, topic
```

Optional columns can include year, marks, unit, subject, etc.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run in Google Colab

```python
!pip install streamlit pyngrok pandas numpy scikit-learn sentence-transformers kagglehub openpyxl
!streamlit run app.py & npx localtunnel --port 8501
```

## Project statement
This project does not predict the exact upcoming exam paper. It recommends important topics based on historical PYQ trends, topic frequency, question similarity, and syllabus mapping.
