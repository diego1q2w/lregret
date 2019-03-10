import os

import pandas as pd

from datasets.linear import LinearProblem
from regresion.linear.feature import PolFeatures
from regresion.linear.linear import LinearRegression


class ComputerHardwareProblem(LinearProblem):

    def dataset_title(self) -> str:
        return "Computer Hardware"

    def plot_dataset(self) -> None:
        """Since there are a lot of attributes printing every graph might be a bit overkilled
        but if you still want to see them feel free to delete this method or even better use this method
        to create your custom graphs for this specific data set"""
        pass

    def __init__(self, regression: LinearRegression, pol_features=PolFeatures(1)) -> None:
        file_name = os.path.join(os.path.dirname(__file__), 'dataset.csv')
        df = pd.read_csv(file_name)

        # one hot encoding
        one_hot_vendor = pd.get_dummies(df['vendorName'])
        df.drop('vendorName', axis=1, inplace=True)

        df = pd.concat([one_hot_vendor, df], axis=1)

        output = df['ERP']
        df.drop('ERP', axis=1, inplace=True)
        df.drop('Model', axis=1, inplace=True)
        super().__init__(df, output, regression, pol_features)


# lr = LinearRegression(learning_rate=0.000000001)
# s = ComputerHardwareProblem(lr)
# s.fit_l1(0.2)
