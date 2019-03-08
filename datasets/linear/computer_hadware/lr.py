# Data set can be found in:
# https://archive.ics.uci.edu/ml/datasets/Computer+Hardware
# Computer Hardware Data Set
# Data Set Information:
#
# The estimated relative performance values were estimated by the authors using a linear regression method.
# See their article (pp 308-313) for more details on how the relative performance values were set.
#
# Attribute Information:
#
# 1. vendor name: 30
# (adviser, amdahl,apollo, basf, bti, burroughs, c.r.d, cambex, cdc, dec,
#  dg, formation, four-phase, gould, honeywell, hp, ibm, ipl, magnuson,
#  microdata, nas, ncr, nixdorf, perkin-elmer, prime, siemens, sperry,
#  sratus, wang)
# 2. Model Name: many unique symbols
# 3. MYCT: machine cycle time in nanoseconds (integer)
# 4. MMIN: minimum main memory in kilobytes (integer)
# 5. MMAX: maximum main memory in kilobytes (integer)
# 6. CACH: cache memory in kilobytes (integer)
# 7. CHMIN: minimum channels in units (integer)
# 8. CHMAX: maximum channels in units (integer)
# 9. PRP: published relative performance (integer)
# 10. ERP: estimated relative performance from the original article (integer)

import pandas as pd

from datasets.linear import LinearProblem
from regresion.linear.linear import LinearRegression


class ComputerHardwareProblem(LinearProblem):

    def dataset_title(self) -> str:
        return "Computer Hardware"

    def plot_dataset(self) -> None:
        """Since there are a lot of attributes printing every graph might be a bit overkilled
        but if you still want to see them feel free to delete this method or even better use this method
        to create your custom graphs for this specific data set"""
        pass

    def __init__(self, regression: LinearRegression) -> None:
        df = pd.read_csv('dataset.csv')

        # one hot encoding
        one_hot_vendor = pd.get_dummies(df['vendorName'])
        df.drop('vendorName', axis=1, inplace=True)

        df = pd.concat([one_hot_vendor, df], axis=1)

        output = df['ERP']
        df.drop('ERP', axis=1, inplace=True)
        df.drop('Model', axis=1, inplace=True)
        super().__init__(df, output, regression)


# lr = LinearRegression(learning_rate=0.000000001)
# s = ComputerHardwareProblem(lr)
# s.fit_l1(0.2)
