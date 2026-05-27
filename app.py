import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="IMDb Movie Intelligence System",
    page_icon="🎬",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("cleaned_imdb_movies.csv")

model = joblib.load("movie_rating_predictor.pkl")

# =========================
# TITLE
# =========================

st.title("🎬 IMDb Movie Intelligence System")

st.markdown("""
This application provides:

✅ Movie Rating Prediction  
✅ Movie Recommendation System  
✅ Interactive Movie Analytics
""")

# =========================
# SIDEBAR
# =========================

option = st.sidebar.selectbox(
    "Choose Feature",
    (
        "Movie Recommendation",
        "Rating Prediction",
        "Dataset Overview"
    )
)

# =========================
# MOVIE RECOMMENDATION
# =========================

if option == "Movie Recommendation":

    st.header("🎥 Movie Recommendation System")

    df['combined_features'] = (
        df['Genre'].astype(str) + ' ' +
        df['Director'].astype(str) + ' ' +
        df['Actor 1'].astype(str) + ' ' +
        df['Actor 2'].astype(str)
    )

    tfidf = TfidfVectorizer(stop_words='english')

    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    cosine_sim = cosine_similarity(tfidf_matrix)

    movie_list = df['Name'].dropna().unique()

    selected_movie = st.selectbox(
        "Select a Movie",
        movie_list
    )

    def recommend_movies(movie_name):

        movie_name = movie_name.lower()

        indices = pd.Series(
            df.index,
            index=df['Name'].str.lower()
        ).drop_duplicates()

        idx = indices[movie_name]

        similarity_scores = list(
            enumerate(cosine_sim[idx])
        )

        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        similarity_scores = similarity_scores[1:6]

        movie_indices = [
            i[0] for i in similarity_scores
        ]

        return df['Name'].iloc[movie_indices]

    if st.button("Recommend"):

        recommendations = recommend_movies(selected_movie)

        st.subheader("Recommended Movies")

        for movie in recommendations:
            st.write("👉", movie)

# =========================
# RATING PREDICTION
# =========================

elif option == "Rating Prediction":

    st.header("⭐ Movie Rating Prediction")

    year = st.number_input(
        "Year",
        1950,
        2026,
        2020
    )

    duration = st.number_input(
        "Duration (minutes)",
        60,
        240,
        120
    )

    votes = st.number_input(
        "Votes",
        100,
        1000000,
        5000
    )

    genre_count = st.slider(
        "Genre Count",
        1,
        5,
        2
    )

    director_frequency = st.slider(
        "Director Popularity",
        1,
        100,
        10
    )

    actor_frequency = st.slider(
        "Actor Popularity",
        1,
        100,
        15
    )

    genre_encoded = st.slider(
        "Genre Encoded",
        0,
        50,
        5
    )

    director_encoded = st.slider(
        "Director Encoded",
        0,
        500,
        20
    )

    if st.button("Predict Rating"):

        movie_age = 2026 - year

        input_data = pd.DataFrame([[
            year,
            duration,
            votes,
            movie_age,
            genre_count,
            director_frequency,
            actor_frequency,
            genre_encoded,
            director_encoded
        ]], columns=[
            'Year',
            'Duration',
            'Votes',
            'Movie_Age',
            'Genre_Count',
            'Director_Frequency',
            'Actor1_Frequency',
            'Genre_Encoded',
            'Director_Encoded'
        ])

        prediction = model.predict(input_data)

        st.success(
            f"Predicted IMDb Rating: {prediction[0]:.2f}"
        )

# =========================
# DATASET OVERVIEW
# =========================

else:

    st.header("📊 Dataset Overview")

    st.write(df.head())

    st.subheader("Dataset Shape")

    st.write(df.shape)

    st.subheader("Rating Statistics")

    st.write(df['Rating'].describe())

    st.subheader("Top Rated Movies")

    top_movies = df.sort_values(
        by='Rating',
        ascending=False
    ).head(10)

    st.dataframe(
        top_movies[['Name', 'Rating']]
    )