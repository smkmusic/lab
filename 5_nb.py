import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

# --------------------------------
# 1. Load dataset from CSV file
# --------------------------------
data = pd.read_csv("td5.csv")

print("Original Dataset:\n")
print(data)

# --------------------------------
# 2. Convert categorical data into numbers
# --------------------------------
encoders = {}

for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    encoders[column] = le

print("\nEncoded Dataset:\n")
print(data)

# --------------------------------
# 3. Split features and target
# --------------------------------
X = data.drop("PlayTennis", axis=1)
y = data["PlayTennis"]

# --------------------------------
# 4. Split dataset into training and testing
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# --------------------------------
# 5. Train Naïve Bayes model
# --------------------------------
model = CategoricalNB()
model.fit(X_train, y_train)

# --------------------------------
# 6. Predict test data
# --------------------------------
y_pred = model.predict(X_test)

print("\nPredicted Values:")
print(y_pred)

print("\nActual Values:")
print(y_test.values)

# --------------------------------
# 7. Calculate Accuracy
# --------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy =", accuracy * 100, "%")

# --------------------------------
# 8. Predict New Sample
# --------------------------------
# Sunny, Cool, High, Strong

sample = pd.DataFrame([[2, 0, 0, 1]], columns=X.columns)

prediction = model.predict(sample)

print("\nNew Prediction:")

if prediction[0] == 1:
    print("Play Tennis = Yes")
else:
    print("Play Tennis = No")