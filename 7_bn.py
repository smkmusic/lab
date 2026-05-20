import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# --------------------------------
# 1. Read dataset
# --------------------------------
data = pd.read_csv("td7.csv")

print("Dataset:\n")
print(data)

# --------------------------------
# 2. Split features and target
# --------------------------------
X = data.iloc[:, :-1]

y = data.iloc[:, -1]

# --------------------------------
# 3. Split training and testing data
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0
)

# --------------------------------
# 4. Create Bayesian model
# --------------------------------
model = GaussianNB()

# --------------------------------
# 5. Train model
# --------------------------------
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
# 7. Accuracy
# --------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy =", accuracy * 100, "%")

# --------------------------------
# 8. Predict new patient
# --------------------------------
# age, sex, cp, trestbps, chol

sample = [[45, 1, 2, 130, 230]]

prediction = model.predict(sample)

print("\nHeart Disease Prediction:")

if prediction[0] == 1:
    print("Patient has Heart Disease")
else:
    print("Patient does NOT have Heart Disease")