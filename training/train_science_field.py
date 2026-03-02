import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# ================= LOAD DATA =================
df = pd.read_csv("datasets/science_dataset.csv")

X_num = df[["logical", "creative", "social", "leadership", "practical"]]
X_text = df["interest_text"]
y = df["career"]

# ================= BERT MODEL =================
bert = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding text using BERT...")
X_text_vec = bert.encode(X_text.tolist(), show_progress_bar=True)

# ================= COMBINE FEATURES =================
X = np.hstack((X_num.values, X_text_vec))

# ================= LABEL ENCODING =================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ================= TRAIN / TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ================= MODEL =================
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

print("Training model...")
model.fit(X_train, y_train)

# ================= EVALUATION =================
y_pred = model.predict(X_test)
print("\nMODEL REPORT:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ================= SAVE MODELS =================
joblib.dump(model, "models/science_field_model.pkl")
joblib.dump(bert, "models/bert_model.pkl")
joblib.dump(le, "models/science_label_encoder.pkl")

print("\nscience_field_model.pkl saved!")
print("bert_model.pkl saved!")
print("science_label_encoder.pkl saved!")