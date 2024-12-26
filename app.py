import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
def load_data():
    file_tsv = pd.read_csv('file.tsv', sep='\t', encoding='utf-8')
    movie_id_titles = pd.read_csv('Movie_Id_Titles.csv')
    movies_metadata = pd.read_csv('movies_metadata.csv', low_memory=False)
    return file_tsv, movie_id_titles, movies_metadata

file_tsv, movie_id_titles, movies_metadata = load_data()

# Page layout
st.title("Movie Recommendation System")
st.sidebar.header("Options")

# Display datasets
if st.sidebar.checkbox("Show Datasets"):
    st.subheader("Uploaded Datasets")
    st.write("**File.tsv**")
    st.dataframe(file_tsv)
    st.write("**Movie Id Titles**")
    st.dataframe(movie_id_titles)
    st.write("**Movies Metadata**")
    st.dataframe(movies_metadata)

# Simple Recommender Function
def simple_recommender(df, metric_column='popularity', n=10):
    df[metric_column] = pd.to_numeric(df[metric_column], errors='coerce')
    top_movies = df.nlargest(n, metric_column)
    return top_movies

# Content-Based Filtering
def content_based_recommender(metadata, movie_title, n=10):
    tfidf = TfidfVectorizer(stop_words='english')
    metadata['overview'] = metadata['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(metadata['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()
    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return metadata.iloc[movie_indices]

# Sidebar options
option = st.sidebar.selectbox(
    "Choose Recommender System:",
    ("Simple Recommender", "Content-Based Filtering")
)

if option == "Simple Recommender":
    st.subheader("Simple Recommender")
    top_n = st.slider("Select number of top movies:", 1, 20, 10)
    result = simple_recommender(movies_metadata, 'popularity', top_n)
    st.write(result)

elif option == "Content-Based Filtering":
    st.subheader("Content-Based Filtering")
    movie_title = st.selectbox("Select a movie title:", movies_metadata['title'].dropna().unique())
    n_recommendations = st.slider("Number of recommendations:", 1, 20, 5)
    if movie_title:
        recommendations = content_based_recommender(movies_metadata, movie_title, n_recommendations)
        st.write(recommendations)

st.sidebar.markdown("---")
st.sidebar.markdown("Developed with ❤️ by [Your Name]")
