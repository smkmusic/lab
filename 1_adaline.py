# Input data
X = [
    [1, 2],
    [2, 3],
    [3, 4]
]

# Actual output
y = [0, 0, 1]

# Initial weights
w1 = 0
w2 = 0

# Bias
b = 0

# Learning rate
lr = 0.1

# Training
for i in range(100):

    for j in range(len(X)):

        # Inputs
        x1 = X[j][0]
        x2 = X[j][1]

        # Prediction
        output = (x1 * w1) + (x2 * w2) + b

        # Error
        error = y[j] - output

        # Update weights
        w1 = w1 + lr * error * x1
        w2 = w2 + lr * error * x2

        # Update bias
        b = b + lr * error

# Print weights
print("Weight 1 =", w1)
print("Weight 2 =", w2)
print("Bias =", b)

print("\nActual and Predicted Outputs")

# Testing
for j in range(len(X)):

    x1 = X[j][0]
    x2 = X[j][1]

    # Predict output
    result = (x1 * w1) + (x2 * w2) + b

    # Threshold
    if result >= 0.5:
        prediction = 1
    else:
        prediction = 0

    # Print outputs
    print("Input =", X[j])
    print("Actual Output =", y[j])
    print("Predicted Output =", prediction)
    print()