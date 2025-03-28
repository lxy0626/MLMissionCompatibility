import pandas as pd
import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
import joblib

# Download stopwords
nltk.download("stopwords")

mission_data = pd.read_csv('collab.csv')
X = mission_data.drop(columns=['compatibility'])
Y = mission_data['compatibility']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=42)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\W", " ", text)  # Remove non-word characters
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])  # Remove stopwords
    return text

print("Columns in X_train:", X_train.columns.tolist())
print("Columns in X_test:", X_test.columns.tolist())

text_column = "mission"  # Change this to the actual column name containing text
X_train[text_column] = X_train[text_column].astype(str).apply(preprocess_text)
X_test[text_column] = X_test[text_column].astype(str).apply(preprocess_text)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train[text_column])  # Ensure column selection
X_test = vectorizer.transform(X_test[text_column])

# Convert sparse matrix to dense array (Optional but useful)
X_train = X_train.toarray()
X_test = X_test.toarray()




best_params = {
    'criterion': 'entropy',
    'max_depth': 10,
    'min_samples_leaf': 1,
    'min_samples_split': 2
}

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, Y_train)

joblib.dump(clf, "decision_tree_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

predictions = clf.predict(X_test)

# âœ… Print Accuracy
print("Accuracy:", accuracy_score(Y_test, predictions))
print(classification_report(Y_test, predictions))    