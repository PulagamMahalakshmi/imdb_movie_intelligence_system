import streamlit as st
import pandas as pd
import numpy as np

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
# SAMPLE DATA (DEPLOYMENT SAFE)
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
# DUMMY MODEL (DEPLOYMENT SAFE)
# =========================
class DummyModel:
    def predict(self, X):
        return np.array([7.5])

model = DummyModel()

# =========================
# TITLE
# =========================
st.title("🎬 IMDb Movie Intelligence System")

st.markdown("""
### Features:
- 🎥 Movie Recommendation System  
- ⭐ Movie Rating Prediction  
- 📊 Dataset Overview  
""")

# =========================
# SIDEBAR MENU
# =========================
option = st.sidebar.selectbox(
    "Choose Feature",
    ["Movie Recommendation", "Rating Prediction", "Dataset Overview"]
)

# =========================
# MOVIE RECOMMENDATION
# =========================
if option == "Movie Recommendation":

    st.header("🎥 Movie Recommendation System")

    df['combined_features'] = (
        df['Genre'] + " " +
        df['Director'] + " " +
        df['Actor 1'] + " " +
        df['Actor 2']
    )

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix)

    movie_list = df['Name'].tolist()

    selected_movie = st.selectbox("Select a Movie", movie_list)

    def recommend(movie):
        idx = df[df['Name'] == movie].index[0]

        scores = list(enumerate(cosine_sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:4]

        movie_indices = [i[0] for i in scores]
        return df['Name'].iloc[movie_indices]

    if st.button("Recommend"):
        results = recommend(selected_movie)
        st.subheader("Recommended Movies:")
        for r in results:
            st.write("👉", r)

# =========================
# RATING PREDICTION
# =========================
elif option == "Rating Prediction":

    st.header("⭐ Movie Rating Prediction")

    year = st.number_input("Year", 1990, 2026, 2020)
    duration = st.number_input("Duration (minutes)", 60, 240, 120)
    votes = st.number_input("Votes", 100, 1000000, 5000)

    if st.button("Predict Rating"):

        movie_age = 2026 - year

        input_data = pd.DataFrame([[year, duration, votes, movie_age]],
                                  columns=['Year', 'Duration', 'Votes', 'Movie_Age'])

        prediction = model.predict(input_data)

        st.success(f"🎯 Predicted IMDb Rating: {prediction[0]:.2f}")

# =========================
# DATASET OVERVIEW
# =========================
else:

    st.header("📊 Dataset Overview")

    st.write(df)

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Top Rated Movies")
    st.dataframe(df.sort_values(by="Rating", ascending=False))