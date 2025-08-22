"""

Linear Regression Implementations

This code demonstrates four approaches to Linear Regression:
1. Simple Linear Rregression using normal equations
2. Linear Regression from scratch using batch gradient descent.
3. Linear Regression from scratch using Stochastic Gradient Descent (SGD).
4. Linear Regression using scikit-learn.

For the from-scratch implementations:
- Weights and bias are initialized to zeros.
- The model is trained using gradient descent to minimize the Mean Squared Error.
- Final learned parameters are returned.
"""


import numpy as np
from sklearn.linear_model import LinearRegression

# Sample Data
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

# -----------------------------
# 1. Simple Linear Rregression using normal equations
# -----------------------------
def LinearReg_simple(X, y):
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    w_bias = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y

    w = w_bias[1:]
    b = w_bias[0]
    return w, b

w_simple, b_simple = LinearReg_simple(X, y)
print("Weights_simple:", w_simple)
print("Bias_simple:", b_simple)

# -----------------------------
# 2. Simple Linear Rregression using batch gradient descent
# -----------------------------
class LinearReg_GD:
    def __init__(self, lr = 0.01, epochs = 1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros((n_features, 1))
        self.b = 0
    
        for i in range(self.epochs):
            y_hat = X @ self.w + self.b

            dw = 1 / n_samples * X.T @ (y_hat - y.reshape(-1, 1))
            db = 1 / n_samples * np.sum(y_hat - y.reshape(-1, 1))
        
            self.w -= self.lr * dw
            self.b -= self.lr * db
        
        return self.w, self.b

model = LinearReg_GD(lr=0.01, epochs=1000)
w, b = model.fit(X, y)
print("Weights_GD:", w)
print("Bias_GD:", b)

# -----------------------------
# 3. Simple Linear Rregression using batch gradient descent
# -----------------------------
class LinearReg_SGD:
    def __init__(self, lr = 0.01, epochs = 1000, batch_size=1, shuffle=True):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros((n_features, 1))
        self.b = 0

        y = y.reshape(-1, 1)

        for epoch in range(self.epochs):
            if self.shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y
        
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                y_hat_batch = X_batch @ self.w + self.b
                
                dw = 1 / X_batch.shape[0] * (X_batch.T @ (y_hat_batch - y_batch) )
                db = 1 / X_batch.shape[0] * np.sum(y_hat_batch - y_batch)
            
                self.w -= self.lr * dw
                self.b -= self.lr * db
        
        return self.w, self.b

model = LinearReg_SGD(lr=0.01, epochs=1000)
w, b = model.fit(X, y)
print("Weights_SGD:", w)
print("Bias_SGD:", b)

# -----------------------------
# 4. Simple Linear Rregression using scikit-learn
# -----------------------------
def LinearReg_sk(X, y):
    model = LinearRegression().fit(X, y)
    w = model.coef_
    b = model.intercept_
    return w, b

w_sk, b_sk = LinearReg_sk(X, y)
print("Weights_sk:", w_sk)
print("Bias_sk:", b_sk)
    
        
