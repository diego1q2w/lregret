import numpy as np
import pandas as pd


class LinearRegression:

    def __init__(self, x: pd.DataFrame, y: pd.Series):
        if type(x) == pd.Series:
            x = pd.DataFrame({'X': x.values})

        x['bias'] = 1
        self.samples_size, self.features_size = x.shape
        self.weight = np.zeros(self.features_size)
        self.costs = np.array([])
        self.samples = x
        self.target = y

    def fit_gradient_descent(self):
        learning_rate = 0.000001
        old_cost = 1000000
        while True:
            y_pred = self.prediction()
            residuals = y_pred - self.target
            gradient_vector = self.samples.T.dot(residuals)
            self.weight -= (learning_rate / self.samples_size) * gradient_vector

            cost = residuals.dot(residuals) / (2 * self.samples_size)
            self.costs = np.append(self.costs, cost)
            if np.abs(old_cost - cost) < 0.001:
                break
            old_cost = cost

    def fit(self) -> None:
        self.weight = np.linalg.solve(self.samples.T.dot(self.samples), self.samples.T.dot(self.target))

    def fit_l2(self, l2: float):
        self.weight = np.linalg.solve(l2*np.eye(len(self.samples.columns)) + self.samples.T.dot(self.samples), self.samples.T.dot(self.target))

    def prediction(self) -> pd.Series:
        return self.samples.dot(self.weight)

    def r2(self) -> float:
        d1 = self.target - self.prediction()
        d2 = self.target - self.target.mean()
        return 1 - d1.dot(d1) / d2.dot(d2)
