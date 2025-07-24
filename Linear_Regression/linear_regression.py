import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numba

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        # Ensure X and y are numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Reshape y if it's 1D
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))  # Changed to column vector
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Compute cost
            cost = (1/(2*n_samples)) * np.sum((y_predicted - y)**2)
            self.cost_history.append(cost)
    
    def predict(self, X):
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X, y):
        y = np.array(y)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

# Example usage
if __name__ == "__main__":
    try:
        # Generate sample data
        np.random.seed(42)
        X = 2 * np.random.rand(100, 1)
        y = 4 + 3 * X + np.random.randn(100, 1)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Our custom implementation
        print("Training our custom implementation...")
        custom_model = LinearRegression(learning_rate=0.01, n_iterations=1000)
        custom_model.fit(X_train, y_train)
        custom_predictions = custom_model.predict(X_test)
        custom_r2 = custom_model.score(X_test, y_test)

        # Scikit-learn implementation
        print("\nTraining scikit-learn implementation...")
        sklearn_model = SklearnLinearRegression()
        sklearn_model.fit(X_train, y_train)
        sklearn_predictions = sklearn_model.predict(X_test)
        sklearn_r2 = r2_score(y_test, sklearn_predictions)

        # Print comparison results
        print("\nResults Comparison:")
        print("Custom Implementation:")
        print(f"Weights: {custom_model.weights[0][0]:.2f}")  # Changed to access first element
        print(f"Bias: {custom_model.bias:.2f}")
        print(f"R² Score: {custom_r2:.4f}")

        print("\nScikit-learn Implementation:")
        print(f"Weights: {sklearn_model.coef_[0][0]:.2f}")
        print(f"Bias: {sklearn_model.intercept_[0]:.2f}")
        print(f"R² Score: {sklearn_r2:.4f}")

        # Plot the results
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.scatter(X_test, y_test, color='blue', label='Test data')
            plt.plot(X_test, custom_predictions, color='red', label='Custom model')
            plt.plot(X_test, sklearn_predictions, color='green', label='Scikit-learn model')
            plt.xlabel('X')
            plt.ylabel('y')
            plt.title('Linear Regression Comparison')
            plt.legend()
            plt.show()
        except ImportError:
            print("Matplotlib not installed. Skipping plot.")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please make sure you have all required packages installed:")
        print("pip install numpy scikit-learn matplotlib")