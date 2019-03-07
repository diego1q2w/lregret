# Data set can be found in:
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/frame.html
# Systolic Blood Pressure Data
#
# The data (X1, X2, X3) are for each patient.
#     X1 = systolic blood pressure
# X2 = age in years
# X3 = weight in pounds

import pandas as pd
import matplotlib.pyplot as plt

from linear_problems import LinearProblem
from regresion.linear.linear import LinearRegression


class SystolicBlood(LinearProblem):

    def __init__(self):
        df = pd.read_excel('dataset.xls')
        self.target = df['X1']
        self.samples = df[['X2', 'X3']]
        self.plot_dataset()

    def plot_dataset(self):
        plt.scatter(self.samples['X2'], self.target)
        plt.scatter(self.samples['X3'], self.target)
        plt.title("Systolic Blood Pressure Data")
        plt.show()

    def print_result(self, fit_type: str):
        print("--- Result ---")
        print("Weights: ", self.regression.weight)
        print("R-squared: ", self.regression.r2())

        plt.plot(self.target, label='Target', c='r')
        plt.plot(self.regression.prediction(), label='Prediction', c='b')
        plt.title(fit_type)
        plt.show()

        plt.plot(self.regression.costs, label='Cost')
        plt.title("Training cost")
        plt.show()

    def fit_l1(self, l1: float):
        pass

    def fit_l2(self, l2: float):
        self.regression = LinearRegression(self.samples, self.target)
        self.regression.fit_l2(l2)
        self.print_result("L2 Regularisation")

    def fit(self):
        self.regression = LinearRegression(self.samples, self.target)
        self.regression.fit_gradient_descent()
        self.print_result("Gradient Decedent")


s = SystolicBlood()
s.fit()
