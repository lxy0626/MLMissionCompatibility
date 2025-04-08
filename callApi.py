'''
pip install fastapi uvicorn
'''

import pandas as pd
import re
import nltk
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nltk.corpus import stopwords

# Download stopwords (only needed once)
nltk.download("stopwords")

# Initialize FastAPI app
app = FastAPI()

# Load trained model & vectorizer
clf = joblib.load("decision_tree_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\W", " ", text)  # Remove non-word characters
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])  # Remove stopwords
    return text

# Define input format
class MissionInput(BaseModel):
    mission: str

# Prediction API endpoint
@app.post("/predict")
def predict(input_data: MissionInput):
    try:
        processed_text = preprocess_text(input_data.mission)
        vectorized_input = vectorizer.transform([processed_text])
        prediction = clf.predict(vectorized_input.toarray())[0]

        return {"mission": input_data.mission, "predicted_compatibility": int(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run API using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)