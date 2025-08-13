import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# -------------------------------
# 1) Load the Iris dataset
# -------------------------------
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# -------------------------------
# 2) One-hot encode labels
# -------------------------------
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# -------------------------------
# 3) Normalize features
# -------------------------------
X = (X - X.mean(axis=0)) / X.std(axis=0)

# -------------------------------
# 4) Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# -------------------------------
# 5) Define network architecture
# -------------------------------
input_size = X_train.shape[1]   # 4 features
hidden_size = 5                 # hidden layer neurons
output_size = y_train.shape[1]  # 3 output neurons
learning_rate = 0.1

# -------------------------------
# 6) Initialize weights and biases
# -------------------------------
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# -------------------------------
# 7) Sigmoid function and derivative
# -------------------------------
def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    """Derivative of sigmoid function"""
    return a * (1 - a)

# -------------------------------
# 8) Training loop
# -------------------------------
epochs = 500
for epoch in range(epochs):
    # ----- Forward Pass -----
    z1 = np.dot(X_train, W1) + b1   # input -> hidden
    a1 = sigmoid(z1)                # hidden activation
    
    z2 = np.dot(a1, W2) + b2         # hidden -> output
    a2 = sigmoid(z2)                 # output activation (sigmoid)
    
    # ----- Loss (Mean Squared Error) -----
    loss = np.mean((y_train - a2) ** 2)
    
    # ----- Backward Pass -----
    # Output layer error
    dz2 = (a2 - y_train) * sigmoid_derivative(a2)
    dW2 = np.dot(a1.T, dz2) / X_train.shape[0]
    db2 = np.sum(dz2, axis=0, keepdims=True) / X_train.shape[0]
    
    # Hidden layer error
    dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(a1)
    dW1 = np.dot(X_train.T, dz1) / X_train.shape[0]
    db1 = np.sum(dz1, axis=0, keepdims=True) / X_train.shape[0]
    
    # ----- Update weights and biases -----
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    
    # Print loss every 50 epochs
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# -------------------------------
# 9) Test the model
# -------------------------------
z1 = np.dot(X_test, W1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, W2) + b2
a2 = sigmoid(z2)

# Convert output to predicted class
predictions = np.argmax(a2, axis=1)
true_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == true_labels) * 100
print(f"Test Accuracy: {accuracy:.2f}%")