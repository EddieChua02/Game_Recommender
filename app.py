import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset
@st.cache
def load_data():
    # Replace with your actual dataset path
    metadata = pd.read_csv("games_metadata.csv")
    return metadata

# Build Recommendation System
def build_recommender(metadata):
    # Ensure dataset has the necessary columns
    metadata['combined_features'] = metadata['description'] + ' ' + metadata['genres']

    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(metadata['combined_features'].fillna(''))

    # Compute cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Index mapping for game names
    indices = pd.Series(metadata.index, index=metadata['name']).drop_duplicates()

    return cosine_sim, indices

# Recommend Games
def recommend_games(game_name, cosine_sim, indices, metadata):
    if game_name not in indices:
        return ["Game not found in the dataset."]

    idx = indices[game_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 recommendations
    game_indices = [i[0] for i in sim_scores]

    return metadata['name'].iloc[game_indices].tolist()

# Streamlit App
st.title("Game Recommender System")
st.sidebar.title("Input")

metadata = load_data()
cosine_sim, indices = build_recommender(metadata)

game_name = st.sidebar.selectbox("Select a game:", metadata['name'].sort_values().unique())

if st.sidebar.button("Recommend"):
    recommendations = recommend_games(game_name, cosine_sim, indices, metadata)
    st.write("### Recommended Games:")
    for rec in recommendations:
        st.write(f"- {rec}")
