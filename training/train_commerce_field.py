import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer

print("Loading dataset...")
df = pd.read_csv("datasets/commerce_dataset.csv")

# Features
X_num = df[["logical", "creative", "social", "leadership", "practical"]].values
texts = df["interest_text"].astype(str).tolist()
y = df["field"]

# Encode labels
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)

print("Loading BERT model...")
bert = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding text using BERT...")
X_text = bert.encode(texts, show_progress_bar=True)

# Combine numeric + BERT embeddings
X = np.hstack((X_num, X_text))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42
)

print("Training model...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nMODEL REPORT:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save models
joblib.dump(model, "models/commerce_field_model.pkl")
joblib.dump(label_encoder, "models/commerce_label_encoder.pkl")
joblib.dump(bert, "models/bert_model.pkl")

print("\ncommerce_field_model.pkl saved!")
print("commerce_label_encoder.pkl saved!")
print("bert_model.pkl saved!")