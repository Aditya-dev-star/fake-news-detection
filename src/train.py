import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from src.preprocess import clean_text


# Load datasets
true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

true_df["label"] = 1
fake_df["label"] = 0

df = pd.concat([true_df, fake_df])
df = df.sample(frac=1, random_state=42)

df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["label"]

# Vectorization
vectorizer = TfidfVectorizer(max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Model
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model & Vectorizer saved successfully.")
