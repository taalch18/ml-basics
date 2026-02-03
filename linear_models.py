import numpy as np
from utils import (
    make_regression_data,
    make_classification_data,
    train_test_split,
    sigmoid,
    mse,
    accuracy,
    set_seed
)


class LinearRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for _ in range(self.epochs):
            y_hat = X @ self.w + self.b
            dw = (1 / n) * X.T @ (y_hat - y)
            db = (1 / n) * np.sum(y_hat - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return X @ self.w + self.b


class LogisticRegressionScratch:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for _ in range(self.epochs):
            z = X @ self.w + self.b
            y_hat = sigmoid(z)

            dw = (1 / n) * X.T @ (y_hat - y)
            db = (1 / n) * np.sum(y_hat - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return (sigmoid(X @ self.w + self.b) >= 0.5).astype(int)


def sklearn_linear(X_train, X_test, y_train):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(X_train, y_train)
    return model.predict(X_test)


def sklearn_logistic(X_train, X_test, y_train):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression().fit(X_train, y_train)
    return model.predict(X_test)


if __name__ == "__main__":
    set_seed(42)

    X, y = make_regression_data(200)
    Xtr, Xte, ytr, yte = train_test_split(X, y)

    lr = LinearRegressionScratch()
    lr.fit(Xtr, ytr)
    print("Linear (scratch) MSE:", mse(yte, lr.predict(Xte)))
    print("Linear (sklearn) MSE:", mse(yte, sklearn_linear(Xtr, Xte, ytr)))

    print("-" * 40)

    X, y = make_classification_data(200)
    Xtr, Xte, ytr, yte = train_test_split(X, y)

    logreg = LogisticRegressionScratch()
    logreg.fit(Xtr, ytr)
    print("Logistic (scratch) Acc:", accuracy(yte, logreg.predict(Xte)))
    print("Logistic (sklearn) Acc:", accuracy(yte, sklearn_logistic(Xtr, Xte, ytr)))
