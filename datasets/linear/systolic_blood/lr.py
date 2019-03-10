import os

import pandas as pd

from datasets.linear import LinearProblem
from regresion.linear.feature import PolFeatures
from regresion.linear.linear import LinearRegression


class SystolicBloodProblem(LinearProblem):

    def __init__(self, regression: LinearRegression, pol_features=PolFeatures(1)) -> None:
        file_name = os.path.join(os.path.dirname(__file__), 'dataset.xls')
        df = pd.read_excel(file_name)
        super().__init__(df[['X2', 'X3']], df['X1'], regression, pol_features)

    def dataset_title(self) -> str:
        return 'Systolic Blood Pressure'


# lr = LinearRegression()
# s = SystolicBloodProblem(lr)
# s.fit()
# s.fit_l2(100000)
