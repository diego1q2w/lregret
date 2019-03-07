from regresion.linear.linear import LinearRegression


class LinearProblem:
    regression = LinearRegression

    def fit(self):
        """ Trains the model using gradient descent
        since it is applicable to every dataset is a required method """
        raise NotImplementedError

    def fit_l2(self, l2: float):
        """ Trains the model using L2 regularisation
         since it is applicable to every dataset is a required method"""
        raise NotImplementedError

    def fit_l1(self, l1: float):
        """ Trains the model using L1 regularisation
         since it is applicable to every dataset is a required method"""
        raise NotImplementedError

    def fit_polynomial(self, degree: int):
        """ Trains the model using a polynomial regression
        since this can be applicable for datasets with one feature it is an optional method"""
        pass
