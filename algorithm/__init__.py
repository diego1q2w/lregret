import numpy as np
import pandas as pd


class LinearRegression:
    def __init__(self, x: pd.DataFrame, y: pd.Series):
        if type(x) == pd.Series:
            x = pd.DataFrame({'X': x.values})

        x['bias'] = 1
        self.samples, self.features = x.shape
        self.w = np.zeros(self.features)
        self.x = x
        self.y = y

    def fit_gradient_descent(self):
        learning_rate = 0.000001
        old_cost = 1000000
        while True:
            y_pred = self.prediction()
            residuals = y_pred - self.y
            gradient_vector = self.x.T.dot(residuals)
            self.w -= (learning_rate / self.samples) * gradient_vector

            cost = residuals.dot(residuals) / (2 * self.samples)
            if np.abs(old_cost - cost) < 0.001:
                break
            old_cost = cost

    def fit(self) -> None:
        self.w = np.linalg.solve(self.x.T.dot(self.x), self.x.T.dot(self.y))

    def fit_l2(self, l2: float):
        self.w = np.linalg.solve(l2*np.eye(len(self.x.columns)) + self.x.T.dot(self.x), self.x.T.dot(self.y))

    def prediction(self) -> pd.Series:
        return self.x.dot(self.w)

    def r2(self) -> float:
        d1 = self.y - self.prediction()
        d2 = self.y - self.y.mean()
        return 1 - d1.dot(d1) / d2.dot(d2)
