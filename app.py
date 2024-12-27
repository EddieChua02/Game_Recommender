import streamlit as st

# Streamlit App
st.title("Game Recommender System")
st.sidebar.title("Input")

# Placeholder for game selection
game_name = st.sidebar.selectbox("Select a game:", ["Game 1", "Game 2", "Game 3"])

# Placeholder for recommendation button
if st.sidebar.button("Recommend"):
    # Display dummy recommendations
    st.write("### Recommended Games:")
    recommendations = ["Recommended Game 1", "Recommended Game 2", "Recommended Game 3"]
    for rec in recommendations:
        st.write(f"- {rec}")