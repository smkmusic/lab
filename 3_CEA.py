import pandas as pd

# Read CSV file
data = pd.read_csv("td3.csv")

print(data)

# Split features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Initialize S and G
S = X[0].copy()

G = [["?" for i in range(len(S))] for i in range(len(S))]

# Candidate Elimination
for i in range(len(X)):

    if y[i] == "Yes":

        for j in range(len(S)):

            if S[j] != X[i][j]:
                S[j] = "?"
                G[j][j] = "?"

    else:

        for j in range(len(S)):

            if X[i][j] != S[j]:
                G[j][j] = S[j]

# Remove empty hypotheses
G_final = [g for g in G if g != ["?"] * len(S)]

print("\nSpecific Hypothesis:")
print(S)

print("\nGeneral Hypothesis:")

for g in G_final:
    print(g)