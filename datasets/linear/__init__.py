import pandas as pd
import matplotlib.pyplot as plt

from regresion.linear.linear import LinearRegression


class LinearProblem:
    def __init__(self, samples: pd.DataFrame, target: pd.Series, regression: LinearRegression) -> None:
        self.samples = samples
        self.target = target
        self.regression = regression
        self.plot_dataset()
        self.regression.set_data(self.samples, self.target)

    def plot_dataset(self) -> None:
        for feature in self.samples.columns:
            plt.scatter(self.samples[feature], self.target)
            plt.title("{} Data".format(self.dataset_title()))
            plt.xlabel(feature, fontsize=18)
            plt.ylabel(self.target.name, fontsize=16)
            plt.show()

    def fit(self) -> None:
        """ Trains the model using gradient descent
        since it is applicable to every dataset is a required method """
        self.regression.fit()
        self.print_result("Gradient Decedent")

    def fit_l2(self, l2: float) -> None:
        """ Trains the model using L2 regularisation
         since it is applicable to every dataset is a required method"""
        self.regression.fit_l2(l2)
        self.print_result("L2 Regularisation")

    def fit_solving(self) -> None:
        """ Linear regression can not only be resolved using gradient descent
         but also by solving the derivate but that depends of the samples matrix, since it needs
         to get the inverse in order to solve the equation x = inv(A)*b it might not be possible
         for all matrix i.e singular matrixes.

         Since we are solving the equation instead of using gradient descent we DO NOT have a costs chart
        """
        self.regression.fit_by_solving()
        self.print_result("Solving the Weights")

    def fit_l1(self, l1: float) -> None:
        """ Trains the model using L1 regularisation
         since it is applicable to every dataset is a required method"""
        pass

    def fit_polynomial(self, degree: int) -> None:
        """ Trains the model using a polynomial regression
        since this can be applicable for datasets with one feature it is an optional method"""
        pass

    def print_result(self,  fit_type: str) -> None:
        print("--- Result for {} Data using {} ---".format(self.dataset_title(), fit_type))
        print("Weights: \n", self.regression.weight)
        print("R-squared: ", self.regression.r2())

        plt.plot(self.target, label='Target')
        plt.plot(self.regression.prediction(), label='Prediction')
        plt.legend()
        plt.show()

        if len(self.regression.costs) > 0:
            plt.plot(self.regression.costs, label='Cost')
            plt.title("Training cost")
            plt.show()
        print('----------------------------\n')

    def dataset_title(self) -> str:
        raise NotImplementedError

