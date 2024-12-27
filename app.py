import streamlit as st

# Streamlit App
st.title("Game Recommender System")
st.sidebar.title("Input")

# User input for the game name
game_name = st.text_input("Enter a game name:")

# Submit button
if st.button("Submit"):
    if game_name:
        # Display dummy recommendations
        st.write(f"### Recommended Games for '{game_name}':")
        recommendations = ["Recommended Game 1", "Recommended Game 2", "Recommended Game 3"]
        for rec in recommendations:
            st.write(f"- {rec}")
    else:
        st.write("Please enter a game name before submitting.")
