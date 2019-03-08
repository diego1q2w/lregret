# Data set can be found in:
#
#
#
# In the following data pairs
# X = national unemployment rate for adult males
# Y = national unemployment rate for adult females
# Reference: Statistical Abstract of the United States
import pandas as pd

from datasets.linear import LinearProblem
from regresion.linear.feature import PolFeatures
from regresion.linear.linear import LinearRegression


class UnemploymentProblem(LinearProblem):

    def __init__(self, regression: LinearRegression) -> None:
        df = pd.read_excel('dataset.xls')
        super().__init__(df['X'], df['Y'], regression)

    def dataset_title(self) -> str:
        return 'National Unemployment'


lr = LinearRegression()
s = UnemploymentProblem(lr)
p_features = PolFeatures(deg=4)
s.fit_polynomial(p_features)
