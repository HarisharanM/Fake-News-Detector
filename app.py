import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")
fake_df["label"] = 0
true_df["label"] = 1
data = pd.concat([fake_df, true_df], ignore_index=True)
data["text"] = data["text"].apply(clean_text)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(data["text"])
y = data["label"]

model = MultinomialNB()
model.fit(X, y)


st.title("📰 Fake News Detection App")
st.subheader("Paste any news article below and see if it's REAL or FAKE")

user_input = st.text_area("Enter News Article Text:")

if st.button("Analyze"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    
    if prediction == 1:
        st.success("✅ This news article appears to be **REAL**.")
    else:
        st.error("⚠️ This news article appears to be **FAKE**.")
