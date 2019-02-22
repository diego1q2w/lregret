import numpy as np
import pandas as pd


class LinearRegression:
    def __init__(self, x: pd.DataFrame, y: pd.Series):
        if type(x) == pd.Series:
            x = pd.DataFrame({'X': x.values})

        self.w = np.array([])
        self.x = x
        self.y = y

    def fit(self) -> None:
        self.x['ones'] = 1
        self.w = np.linalg.solve(self.x.T.dot(self.x), self.x.T.dot(self.y))

    def prediction(self) -> pd.Series:
        return self.x.dot(self.w)

    def r2(self) -> float:
        d1 = self.y - self.prediction()
        d2 = self.y - self.y.mean()
        return 1 - d1.dot(d1) / d2.dot(d2)
