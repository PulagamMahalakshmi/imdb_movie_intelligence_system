import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

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
# IMDb DARK THEME
# =========================
st.markdown("""
<style>

.stApp {
    background-color: #111111;
    color: #FFFFFF;
}

h1, h2, h3 {
    color: #FFD700;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1E1E1E;
}

/* Buttons */
.stButton>button {
    background-color: #FFD700;
    color: black;
    border-radius: 10px;
    font-weight: bold;
    border: none;
    padding: 0.5rem 1rem;
}

.stButton>button:hover {
    background-color: #FFC107;
    color: black;
}

/* Text Inputs */
input, textarea {
    color: white !important;
    background-color: #2b2b2b !important;
}

/* Select Box */
div[data-baseweb="select"] {
    color: black;
}

/* Radio buttons */
.stRadio label {
    color: white !important;
}

/* Slider text */
.stSlider label {
    color: white !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: white;
}

/* Markdown text */
p, label, div {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# Selectbox text color fix
st.markdown("""
<style>

/* Selectbox selected value */
div[data-baseweb="select"] > div {
    color: black !important;
    background-color: white !important;
}

/* Dropdown menu */
ul {
    background-color: white !important;
}

li {
    color: black !important;
}

</style>
""", unsafe_allow_html=True)
# =========================
# DROPDOWN FIX
# =========================
st.markdown("""
<style>

/* Selectbox main area */
div[data-baseweb="select"] > div {
    background-color: white !important;
    color: black !important;
}

/* Selected text */
div[data-baseweb="select"] span {
    color: black !important;
}

/* Dropdown options */
li[role="option"] {
    background-color: white !important;
    color: black !important;
}

/* Hover effect */
li[role="option"]:hover {
    background-color: #FFD700 !important;
    color: black !important;
}

</style>
""", unsafe_allow_html=True)
# =========================
# LOAD MODEL
# =========================
model_path = os.path.join(
    os.path.dirname(__file__),
    "movie_rating_predictor.pkl"
)

model = joblib.load(model_path)

# =========================
# DATA
# =========================
df = pd.DataFrame({
    "Name": [
        "Avatar",
        "Titanic",
        "Inception",
        "Interstellar",
        "Joker"
    ],

    "Genre": [
        "Action",
        "Romance",
        "Sci-Fi",
        "Sci-Fi",
        "Drama"
    ],

    "Director": [
        "James Cameron",
        "James Cameron",
        "Christopher Nolan",
        "Christopher Nolan",
        "Todd Phillips"
    ],

    "Actor 1": [
        "Sam Worthington",
        "Leonardo DiCaprio",
        "Leonardo DiCaprio",
        "Matthew McConaughey",
        "Joaquin Phoenix"
    ],

    "Actor 2": [
        "Zoe Saldana",
        "Kate Winslet",
        "Joseph Gordon-Levitt",
        "Anne Hathaway",
        "Robert De Niro"
    ],

    "Rating": [
        7.8,
        7.9,
        8.8,
        8.6,
        8.4
    ]
})

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🎬 Navigation")

option = st.sidebar.radio(
    "Select Feature",
    [
        "Movie Recommendation",
        "Rating Prediction",
        "Dataset Overview"
    ]
)

# =========================
# BUILD SIMILARITY
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

    tfidf = TfidfVectorizer(
        stop_words="english"
    )

    matrix = tfidf.fit_transform(
        temp["features"]
    )

    similarity = cosine_similarity(matrix)

    return similarity

cosine_sim = build_similarity(df)

# =========================
# RECOMMEND FUNCTION
# =========================
def recommend(movie_name):

    if movie_name not in df["Name"].values:
        return []

    idx = df[df["Name"] == movie_name].index[0]

    scores = list(
        enumerate(cosine_sim[idx])
    )

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )[1:4]

    movie_indices = [
        i[0] for i in scores
    ]

    return df["Name"].iloc[movie_indices]

# =========================
# TITLE
# =========================
st.title("🎬 IMDb Movie Intelligence System")

st.markdown("""
### AI-powered Movie Recommendation & Rating Prediction
""")

# =========================
# MOVIE RECOMMENDATION
# =========================
if option == "Movie Recommendation":

    st.header("🎥 Movie Recommendation System")

    selected_movie = st.selectbox(
        "Select a Movie",
        df["Name"].tolist()
    )

    if st.button("Recommend Movies"):

        results = recommend(selected_movie)

        if len(results) > 0:

            st.subheader("Recommended Movies")

            for movie in results:
                st.success(f"👉 {movie}")

        else:
            st.warning(
                "No recommendations found"
            )

# =========================
# RATING PREDICTION
# =========================
elif option == "Rating Prediction":

    st.header("⭐ Movie Rating Prediction")

    year = st.number_input(
        "Year",
        1990,
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

    director_freq = st.slider(
        "Director Frequency",
        1,
        100,
        10
    )

    actor_freq = st.slider(
        "Actor Frequency",
        1,
        100,
        10
    )

    genre_enc = st.slider(
        "Genre Encoded",
        0,
        50,
        5
    )

    director_enc = st.slider(
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

        prediction = model.predict(
            input_data
        )

        st.success(
            f"🎯 Predicted IMDb Rating: {prediction[0]:.2f}"
        )

# =========================
# DATASET OVERVIEW
# =========================
else:

    st.header("📊 Dataset Overview")

    st.dataframe(df)

    st.subheader("Dataset Shape")

    st.write(df.shape)

    # =========================
    # TOP MOVIES CHART
    # =========================
    st.subheader("🏆 Top Rated Movies")

    top_movies = df.sort_values(
        by="Rating",
        ascending=False
    )

    fig1 = px.bar(
        top_movies,
        x="Name",
        y="Rating",
        title="Top Rated Movies"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # =========================
    # GENRE DISTRIBUTION
    # =========================
    st.subheader("🎭 Genre Distribution")

    genre_counts = df["Genre"].value_counts()

    fig2 = px.pie(
        names=genre_counts.index,
        values=genre_counts.values,
        title="Genre Distribution"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # =========================
    # RATING DISTRIBUTION
    # =========================
    st.subheader("⭐ Ratings Distribution")

    fig3 = px.histogram(
        df,
        x="Rating",
        nbins=5,
        title="Ratings Distribution"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )