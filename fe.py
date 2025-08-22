

class LinearRegression_sc:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros((n_features, 1))
        self.b = 0

        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.w) + self.b
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y.reshape(-1, 1)))
            db = (1 / n_samples) * np.sum(y_predicted - y.reshape(-1, 1))

            # Update weights and bias
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
        
        return self.w, self.b

model = LinearRegression_sc(learning_rate=0.01, n_iterations=1000)
w, b = model.fit(X, y)
print("Weights:", w)
print("Bias:", b)
 
class LinearRegression: