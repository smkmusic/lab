import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv("td4.csv")

print("\nDataset:\n")
print(df)

# convert text data to numbers (fresh encoder per column)
for column in df.columns:
    df[column] = LabelEncoder().fit_transform(df[column])

X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

plt.figure(figsize=(12, 8))
plot_tree(model,
          feature_names=X.columns,
          class_names=model.classes_.astype(str),
          filled=True)

plt.title("Decision Tree using ID3")
plt.show()