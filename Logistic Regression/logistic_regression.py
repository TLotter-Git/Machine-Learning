import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Compute cost (log loss)
            cost = -(1 / n_samples) * np.sum(
                y * np.log(y_predicted + 1e-15) + (1 - y) * np.log(1 - y_predicted + 1e-15)
            )
            self.cost_history.append(cost)

    def predict_proba(self, X):
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

# Example usage
if __name__ == "__main__":
    try:
        # Generate synthetic binary classification data
        np.random.seed(42)
        X = np.random.randn(200, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Custom implementation
        print("Training our custom implementation...")
        custom_model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
        custom_model.fit(X_train, y_train)
        custom_predictions = custom_model.predict(X_test)
        custom_acc = custom_model.score(X_test, y_test)

        # Scikit-learn implementation
        print("\nTraining scikit-learn implementation...")
        sklearn_model = SklearnLogisticRegression()
        sklearn_model.fit(X_train, y_train)
        sklearn_predictions = sklearn_model.predict(X_test)
        sklearn_acc = accuracy_score(y_test, sklearn_predictions)

        # Print comparison results
        print("\nResults Comparison:")
        print("Custom Implementation:")
        print(f"Weights: {custom_model.weights.ravel()}")
        print(f"Bias: {custom_model.bias:.2f}")
        print(f"Accuracy: {custom_acc:.4f}")

        print("\nScikit-learn Implementation:")
        print(f"Weights: {sklearn_model.coef_.ravel()}")
        print(f"Bias: {sklearn_model.intercept_[0]:.2f}")
        print(f"Accuracy: {sklearn_acc:.4f}")

        # Plot the results
        try:
            import matplotlib.pyplot as plt
            # Plot test data
            plt.figure(figsize=(10, 6))
            plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', edgecolor='k', label='Test data')

            # Create a grid to plot decision boundaries
            x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
            y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
            grid = np.c_[xx.ravel(), yy.ravel()]

            # Custom model decision boundary
            zz_custom = custom_model.predict(grid).reshape(xx.shape)
            plt.contour(xx, yy, zz_custom, levels=[0.5], colors='red', linewidths=2, linestyles='--', label='Custom boundary')

            # Sklearn model decision boundary
            zz_sklearn = sklearn_model.predict(grid).reshape(xx.shape)
            plt.contour(xx, yy, zz_sklearn, levels=[0.5], colors='green', linewidths=2, linestyles='-', label='Sklearn boundary')

            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('Logistic Regression Decision Boundary')
            red_patch = plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label='Custom boundary')
            green_patch = plt.Line2D([0], [0], color='green', lw=2, linestyle='-', label='Sklearn boundary')
            plt.legend(handles=[red_patch, green_patch])
            plt.show()
        except ImportError:
            print("Matplotlib not installed. Skipping plot.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please make sure you have all required packages installed:")
        print("pip install numpy scikit-learn")