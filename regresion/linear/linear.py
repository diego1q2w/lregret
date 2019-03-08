import numpy as np
import pandas as pd
from typing import Callable

from datasets import InfinitError


class LinearRegression:
    samples = pd.DataFrame
    target = pd.Series

    def __init__(self, learning_rate=0.000001) -> None:
        self.samples_size, self.features_size = (0, 0)
        self.weight = np.zeros(self.features_size)
        self.costs = np.array([])

        self.learning_rate = learning_rate

    def set_data(self, samples: pd.DataFrame, target: pd.Series) -> None:
        self.samples = samples
        self.samples['bias'] = 1

        self.samples_size, self.features_size = self.samples.shape
        self.target = target

    def _gradient_descent(self, func: Callable[[pd.Series], pd.Series]) -> None:
        old_cost = 1000000
        while True:
            residuals = self.prediction() - self.target
            gradient_vector = func(residuals)
            self.weight -= (self.learning_rate / self.samples_size) * gradient_vector.astype('float')

            cost = residuals.dot(residuals) / (2 * self.samples_size)
            self.costs = np.append(self.costs, cost)

            if np.math.isinf(cost):
                raise InfinitError("Gradient descent reach a cost of inf")

            if np.abs(old_cost - cost) < 0.001:
                break

            old_cost = cost

    def fit(self) -> None:
        self.weight = np.zeros(self.features_size)
        self.costs = np.array([])

        def func(residuals: pd.Series) -> pd.Series:
            return self.samples.T.dot(residuals)

        self._gradient_descent(func)

    def fit_l2(self, l2: float) -> None:
        self.weight = np.zeros(self.features_size)
        self.costs = np.array([])

        def func(residuals: pd.Series) -> pd.Series:
            return self.samples.T.dot(residuals) + l2 * self.weight.astype('float')

        self._gradient_descent(func)

    def fit_l1(self, l1: float) -> None:
        self.weight = np.zeros(self.features_size)
        self.costs = np.array([])

        def func(residuals: pd.Series) -> pd.Series:
            return self.samples.T.dot(residuals) + l1 * np.sign(self.weight)

        self._gradient_descent(func)

    def fit_by_solving(self) -> None:
        self.costs = np.array([])
        self.weight = np.linalg.solve(self.samples.T.dot(self.samples), self.samples.T.dot(self.target))

    def prediction(self) -> pd.Series:
        return self.samples.dot(self.weight)

    def r2(self) -> float:
        d1 = self.target - self.prediction()
        d2 = self.target - self.target.mean()
        return 1 - d1.dot(d1) / d2.dot(d2)
