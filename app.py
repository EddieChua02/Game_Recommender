import streamlit as st
import pandas as pd
import joblib
from scipy.sparse import load_npz

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("steam.csv")  # Replace with your dataset path

# Load Precomputed Sparse Model
@st.cache_data
def load_model():
    sparse_cosine_sim = joblib.load('cosine_sim_sparse.joblib')  # Load sparse matrix
    indices = joblib.load('indices.joblib')  # Load indices
    return sparse_cosine_sim, indices

# Recommend Games
def recommend_games(game_name, cosine_sim, indices, metadata):
    if game_name not in indices:
        return ["Game not found in the dataset."]
    
    idx = indices[game_name]
    sim_scores = cosine_sim[idx].toarray().flatten()
    sim_scores = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 recommendations
    game_indices = [i[0] for i in sim_scores]
    
    return metadata['name'].iloc[game_indices].tolist()

# Streamlit App
st.title("Game Recommender System")
st.sidebar.title("Input")

metadata = load_data()
cosine_sim, indices = load_model()

game_name = st.sidebar.selectbox("Select a game:", metadata['name'].sort_values().unique())

if st.sidebar.button("Recommend"):
    recommendations = recommend_games(game_name, cosine_sim, indices, metadata)
    st.write("### Recommended Games:")
    for rec in recommendations:
        st.write(f"- {rec}")
