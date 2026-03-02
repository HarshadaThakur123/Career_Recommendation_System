import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("datasets/synthetic_student_data.csv")

num_features = df[[
    "maths8","science8","english8","history8","geography8",
    "maths9","science9","english9","history9","geography9",
    "maths10","science10","english10","history10","geography10",
    "logical","creative","social","leadership","practical"
]]

text_data = df["interest_text"].tolist()
y = df["stream"]

# BERT model
bert = SentenceTransformer("all-MiniLM-L6-v2")
text_embeddings = bert.encode(text_data)

# Combine numeric + text features
X = np.hstack((num_features.values, text_embeddings))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=200)
lr = LogisticRegression(max_iter=1000)
xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")

ensemble = VotingClassifier(
    estimators=[("rf", rf), ("lr", lr), ("xgb", xgb)],
    voting="soft"
)

print("Training model...")
ensemble.fit(X_train, y_train)

# ================= EVALUATION =================
y_pred = ensemble.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nMODEL ACCURACY:", round(accuracy * 100, 2), "%")
print("\nCONFUSION MATRIX:\n", cm)
print("\nCLASSIFICATION REPORT:\n", report)

# ================= SAVE MODEL =================
joblib.dump(ensemble, "models/career_model.pkl")
joblib.dump(bert, "models/bert_model.pkl")

print("\n🎯 Ensemble BERT model trained and saved successfully")