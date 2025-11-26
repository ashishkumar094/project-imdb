import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# 📌 Load IMDB dataset (from HuggingFace or local CSV)
df = pd.read_csv(r"C:\Users\ashis\OneDrive\Desktop\Python\Dataset\IMDB Dataset.csv")   # columns: review, sentiment

# Convert labels to numeric
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.2, random_state=42
)

# 📌 Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression(max_iter=200))
])

# Train
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model
joblib.dump(model, "sentiment_model.pkl")
print("Model saved as sentiment_model.pkl")
