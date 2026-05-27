import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("cleaned_imdb_movies.csv")

# Feature selection (must match your dataset)
features = [
    "Year",
    "Duration",
    "Votes",
    "Movie_Age",
    "Genre_Count",
    "Director_Frequency",
    "Actor1_Frequency",
    "Genre_Encoded",
    "Director_Encoded"
]

X = df[features]
y = df["Rating"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "movie_rating_predictor.pkl")

print("Model trained and saved successfully!")