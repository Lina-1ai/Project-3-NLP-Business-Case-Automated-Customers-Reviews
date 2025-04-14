import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertModel
from transformers import pipeline
import torch
from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
import joblib

# Load environment variables
from openai import OpenAI
api_key = st.secrets['OPENAI_API']
client = OpenAI(api_key=api_key)

# Load tokenizer and models
try:
    # Classification
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    clf_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    clf = pipeline("text-classification", model=clf_model, tokenizer=tokenizer)

    # Embedding model
    embedding_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    # Load KMeans model (make sure the .pkl file is in the same directory)
    kmeans = joblib.load("kmeans_model.pkl")

except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    clf = None
    embedding_model = None
    kmeans = None

# --- Classification Function ---
def classify_review(review_text):
    try:
        result = clf(review_text)
        label = result[0]['label']
        score = result[0]['score']
        return label, score
    except Exception as e:
        st.error(f"Error classifying review: {str(e)}")
        return "Unknown", 0.0

# --- Embedding + Clustering Function ---
def get_cluster(review_text):
    try:
        inputs = tokenizer(review_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = embedding_model(**inputs)
        
        # Extract and convert embedding to float64
        embedding = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
        embedding = np.array(embedding, dtype=np.float64).squeeze()
        print("Embedding dtype:", embedding.dtype)  # Debug
        
        # Get cluster index
        cluster_id = kmeans.predict([embedding])[0]
        
        # Map cluster ID to category name
        cluster_names = {
            0: 'Fire tablets',
            1: 'Kindle',
            2: 'Audio and video',
            3: 'Fire tablets for kids'
        }
        
        cluster_name = cluster_names.get(cluster_id, "Unknown")
        
        return cluster_name
    except Exception as e:
        st.error(f"Error getting cluster: {str(e)}")
        return "Unknown"

# --- OpenAI Analysis Function ---
def analyze_with_openai(review_text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates product category analysis."},
                {"role": "user", "content": f"""Based on the following product review, generate a short article that includes:
1. Top 3 products in this category and key differences between them
2. Top complaints for each of those products
3. Worst product in the category and why it should be avoided

Review: {review_text}"""}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error analyzing with OpenAI: {str(e)}")
        return "Unable to analyze review with OpenAI."


# --- Streamlit Interface ---
# Adding CSS to modify the background and title color
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5; /* Light Gray background */
        }
        .title {
            color: orange; /* Title color */
        }
        h1 {
            color: orange; /* Title color */
        }
        .stTextInput>div>div>input {
            background-color: #ffffff; /* White background for text input */
        }
        .stButton>button {
            background-color: #ff6600; /* Orange buttons */
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Add title with customized color
st.title("Customer Review Classifier and Analyzer 77")
st.write("Enter a product review, and the application will classify it, cluster it, and analyze it using OpenAI GPT. 😊")

user_review = st.text_area("Enter the product review:")

if user_review:
    if clf is not None:
        sentiment, confidence = classify_review(user_review)
        sentiment_emoji = "😃" if sentiment == "POSITIVE" else "😞"  # Add emoji based on sentiment
        st.write(f"**Sentiment Classification:** {sentiment_emoji} {sentiment} (Confidence: {confidence:.2f})")

    if kmeans is not None:
        cluster_name = get_cluster(user_review)
        cluster_emoji = "📱" if "tablet" in cluster_name.lower() else "💻"  # Add emoji for cluster
        st.write(f"**KMeans Cluster:** {cluster_emoji} {cluster_name}")

    gpt_analysis = analyze_with_openai(user_review)
    st.write("**OpenAI GPT Analysis:** 📝")
    st.write(gpt_analysis)
