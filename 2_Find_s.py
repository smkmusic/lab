import pandas as pd

# Read CSV file
data = pd.read_csv("td2.csv")

print(data)

# Split features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Initialize hypothesis
h = X[0].copy()

# Find-S Algorithm
for i in range(len(X)):

    if y[i] == "Yes":

        for j in range(len(h)):

            if h[j] != X[i][j]:
                h[j] = "?"

print("\nMost Specific Hypothesis:")
print(h)