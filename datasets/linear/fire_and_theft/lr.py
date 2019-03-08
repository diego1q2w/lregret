import pandas as pd

from datasets.linear import LinearProblem
from regresion.linear.feature import PolFeatures
from regresion.linear.linear import LinearRegression


class FireAndTheftProblem(LinearProblem):

    def __init__(self, regression: LinearRegression) -> None:
        df = pd.read_excel('dataset.xls')
        super().__init__(df['X'], df['Y'], regression)

    def dataset_title(self) -> str:
        return 'Fire and Theft in Chicago'


# lr = LinearRegression()
# s = FireAndTheftProblem(lr)
# p_features = PolFeatures(deg=4)
# s.fit_polynomial(p_features)