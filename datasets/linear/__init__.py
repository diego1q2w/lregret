import os

import pandas as pd


from regresion.linear.feature import PolFeatures
from regresion.linear.linear import LinearRegression

from matplotlib import pyplot as plt


class LinearProblem:
    def __init__(self,
                 samples: pd.DataFrame,
                 target: pd.Series,
                 regression: LinearRegression,
                 pol_features=PolFeatures(1)) -> None:
        if type(samples) == pd.Series:
            self.samples = pd.DataFrame({'X': samples.values})
        else:
            self.samples = samples

        self.target = target
        self.regression = regression
        self.pol_features = pol_features
        self.plot_dataset()

        self.samples = pol_features.generate_pol(self.samples)
        self.regression.set_data(self.samples, self.target)

    def plot_dataset(self) -> None:
        for feature in self.samples.columns:
            fig = plt.figure()

            plt.scatter(self.samples[feature], self.target)
            plt.title("{} Data Set".format(self.dataset_title()))
            plt.xlabel(feature, fontsize=18)
            plt.ylabel(self.target.name, fontsize=16)

            file_name = 'dataset_{}_vs_{}.png'.format(feature, self.target.name)
            self.show_plot(file_name, fig)

    def show_plot(self, file_name: str, fig: plt.Figure):
        is_docker = os.getenv('SAVE_INTO_FILE', False)
        if is_docker:
            file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'tmp_figures', file_name)
            fig.savefig(file_path, dpi=fig.dpi)
        else:
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
        self.regression.fit_l1(l1)
        self.print_result("L1 Regularisation")

    def print_result(self,  fit_type: str) -> None:
        print("\n--- Result for {} Data using {} ---".format(self.dataset_title(), fit_type))
        print("Weights: \n", self.regression.weight)
        print("R-squared: ", self.regression.r2())

        fig = plt.figure()
        plt.plot(self.target, label='Target')
        plt.plot(self.regression.prediction(), label='Prediction')
        plt.legend()
        file_name = 'target_vs_prediction.png'
        self.show_plot(file_name, fig)

        if len(self.regression.costs) > 0:
            fig = plt.figure()
            plt.plot(self.regression.costs, label='Cost')
            plt.title("Training cost")
            file_name = 'cost.png'
            self.show_plot(file_name, fig)
        print('----------------------------\n')

    def dataset_title(self) -> str:
        raise NotImplementedError

