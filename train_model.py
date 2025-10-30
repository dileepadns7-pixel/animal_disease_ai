import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("real_animal_disease_dataset.csv")
df['text'] = df['species'] + ' ' + df['symptoms']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
le = LabelEncoder()
y = le.fit_transform(df['disease'])

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X, y)

joblib.dump(model, "real_animal_disease_model.joblib")
joblib.dump(le, "label_encoder.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")

print("✅ Model trained and saved.")
