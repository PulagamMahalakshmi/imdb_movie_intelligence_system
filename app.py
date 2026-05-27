import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PAGE CONFIG (MUST BE FIRST)
# =========================
st.set_page_config(
    page_title="IMDb Movie Intelligence System",
    page_icon="🎬",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
model_path = os.path.join(os.path.dirname(__file__), "movie_rating_predictor.pkl")
model = joblib.load(model_path)

# =========================
# DATA (DEMO / SAFE)
# =========================
df = pd.DataFrame({
    "Name": ["Avatar", "Titanic", "Inception", "Interstellar", "Joker"],
    "Genre": ["Action", "Romance", "Sci-Fi", "Sci-Fi", "Drama"],
    "Director": ["James Cameron", "James Cameron", "Christopher Nolan", "Christopher Nolan", "Todd Phillips"],
    "Actor 1": ["Sam Worthington", "Leonardo DiCaprio", "Leonardo DiCaprio", "Matthew McConaughey", "Joaquin Phoenix"],
    "Actor 2": ["Zoe Saldana", "Kate Winslet", "Joseph Gordon-Levitt", "Anne Hathaway", "Robert De Niro"],
    "Rating": [7.8, 7.9, 8.8, 8.6, 8.4]
})

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🎬 Navigation")

option = st.sidebar.radio(
    "Select Feature",
    ["Movie Recommendation", "Rating Prediction", "Dataset Overview"]
)

# =========================
# SIMILARITY MODEL (CACHED)
# =========================
@st.cache_data
def build_similarity(data):
    temp = data.copy()

    temp["features"] = (
        temp["Genre"] + " " +
        temp["Director"] + " " +
        temp["Actor 1"] + " " +
        temp["Actor 2"]
    )

    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(temp["features"])
    return cosine_similarity(matrix)

cosine_sim = build_similarity(df)

# =========================
# RECOMMEND FUNCTION
# =========================
def recommend(movie_name):
    if movie_name not in df["Name"].values:
        return []

    idx = df[df["Name"] == movie_name].index[0]

    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:4]

    movie_indices = [i[0] for i in scores]
    return df["Name"].iloc[movie_indices]

# =========================
# TITLE
# =========================
st.title("🎬 IMDb Movie Intelligence System")
st.markdown("AI-powered Movie Recommendation & Rating Prediction System")

# =========================
# MOVIE RECOMMENDATION
# =========================
if option == "Movie Recommendation":

    st.header("🎥 Movie Recommendation System")

    selected_movie = st.selectbox("Select a Movie", df["Name"].tolist())

    if st.button("Recommend Movies"):

        results = recommend(selected_movie)

        if len(results) > 0:
            st.subheader("Recommended Movies")
            for movie in results:
                st.success(movie)
        else:
            st.warning("No recommendations found")

# =========================
# RATING PREDICTION
# =========================
elif option == "Rating Prediction":

    st.header("⭐ Movie Rating Prediction")

    year = st.number_input("Year", 1990, 2026, 2020)
    duration = st.number_input("Duration (minutes)", 60, 240, 120)
    votes = st.number_input("Votes", 100, 1000000, 5000)
    genre_count = st.slider("Genre Count", 1, 5, 2)
    director_freq = st.slider("Director Frequency", 1, 100, 10)
    actor_freq = st.slider("Actor Frequency", 1, 100, 10)
    genre_enc = st.slider("Genre Encoded", 0, 50, 5)
    director_enc = st.slider("Director Encoded", 0, 500, 20)

    if st.button("Predict Rating"):

        movie_age = 2026 - year

        input_data = pd.DataFrame([[ 
            year,
            duration,
            votes,
            movie_age,
            genre_count,
            director_freq,
            actor_freq,
            genre_enc,
            director_enc
        ]], columns=[
            "Year",
            "Duration",
            "Votes",
            "Movie_Age",
            "Genre_Count",
            "Director_Frequency",
            "Actor1_Frequency",
            "Genre_Encoded",
            "Director_Encoded"
        ])

        prediction = model.predict(input_data)

        st.success(f"🎯 Predicted IMDb Rating: {prediction[0]:.2f}")

# =========================
# DATASET OVERVIEW
# =========================
else:

    st.header("📊 Dataset Overview")

    st.dataframe(df)

    st.write("Shape:", df.shape)

    st.subheader("Top Rated Movies")
    st.dataframe(df.sort_values(by="Rating", ascending=False))