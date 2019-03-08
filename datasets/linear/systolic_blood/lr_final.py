import pandas as pd
import matplotlib.pyplot as plt

from datasets.linear import LinearProblem


class SystolicBlood(LinearProblem):

    def __init__(self):
        df = pd.read_excel('dataset.xls')
        super().__init__(df[['X2', 'X3']], df['X1'])

    def dataset_title(self) -> str:
        return 'Systolic Blood Pressure'


s = SystolicBlood()
s.fit()
# s.fit_l2(100000)
