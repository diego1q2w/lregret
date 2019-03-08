import pandas as pd

from datasets.linear import LinearProblem
from regresion.linear.linear import LinearRegression


class SystolicBloodProblem(LinearProblem):

    def __init__(self, regression: LinearRegression) -> None:
        df = pd.read_excel('dataset.xls')
        super().__init__(df[['X2', 'X3']], df['X1'], regression)

    def dataset_title(self) -> str:
        return 'Systolic Blood Pressure'


# lr = LinearRegression()
# s = SystolicBlood(lr)
# s.fit()
# s.fit_l2(100000)
