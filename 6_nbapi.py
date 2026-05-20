import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

# --------------------------------
# 1. Load dataset from CSV file
# --------------------------------
df = pd.read_csv("td6.csv")

print("Dataset:\n")
print(df)

# --------------------------------
# 2. Split features and labels
# --------------------------------
X = df["text"]
y = df["label"]

# --------------------------------
# 3. Convert text into numeric vectors
# --------------------------------
vectorizer = CountVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# --------------------------------
# 4. Split training and testing data
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.3,
    random_state=42
)

# --------------------------------
# 5. Train Naïve Bayes classifier
# --------------------------------
model = MultinomialNB()

model.fit(X_train, y_train)

# --------------------------------
# 6. Predict test data
# --------------------------------
y_pred = model.predict(X_test)

print("\nPredicted Labels:")
print(y_pred)

# --------------------------------
# 7. Accuracy, Precision, Recall
# --------------------------------
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average='macro'
)

recall = recall_score(
    y_test,
    y_pred,
    average='macro'
)

print("\nAccuracy :", accuracy)

print("Precision:", precision)

print("Recall   :", recall)

# --------------------------------
# 8. Detailed Report
# --------------------------------
print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# --------------------------------
# 9. Predict new documents
# --------------------------------
new_docs = [
    "Machine learning is the future",
    "Football players are practicing"
]

new_docs_vectorized = vectorizer.transform(new_docs)

predictions = model.predict(new_docs_vectorized)

print("\nNew Document Predictions:\n")

for i in range(len(new_docs)):
    print(new_docs[i], " --> ", predictions[i])