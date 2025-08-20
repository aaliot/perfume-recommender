import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data  # caches data for faster reloads
def load_data():
    df = pd.read_csv("data/fra_cleaned.csv", sep=";", encoding="windows-1252")
    return df

df = load_data()
st.title("Perfume Recommender & Clustering")
st.write("Explore perfumes by scent notes, clusters, and recommendations.")

df['All_Notes'] = df['Top'].fillna('') + ', ' + df['Middle'].fillna('') + ', ' + df['Base'].fillna('')

perfume_list = df['Perfume'].tolist()
selected_perfume = st.selectbox("Choose a perfume:", perfume_list)


vectorizer = CountVectorizer()
X_notes = vectorizer.fit_transform(df['All_Notes'])
similarity_matrix = cosine_similarity(X_notes)

import numpy as np
def recommend_perfume(name, top_n=5):
    idx = df[df['Perfume'] == name].index[0]
    similar_indices = np.argsort(-similarity_matrix[idx])[1:top_n+1]
    return df.iloc[similar_indices][['Perfume','Brand','All_Notes']]

st.subheader("Recommended perfumes:")
recommendations = recommend_perfume(selected_perfume)
st.dataframe(recommendations)
