import numpy as np
import pandas as pd


class PolFeatures:

    def __init__(self, deg: int):
        self.deg = deg

    def generate_pol(self, x: pd.Series) -> pd.DataFrame:
        pol = {}
        x_mult = pd.Series(np.ones(len(x)))
        for i in range(self.deg):
            key = 'X{}'.format(i+1)
            x_mult *= x
            pol[key] = x_mult.values

        return pd.DataFrame(pol)
