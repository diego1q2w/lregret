import numpy as np
import pandas as pd


class PolFeatures:

    def __init__(self, deg: int):
        self.deg = deg

    def generate_pol(self, samples: pd.DataFrame) -> pd.DataFrame:
        if type(samples) == pd.Series:
            samples = pd.DataFrame({'X': samples.values})
        else:
            samples = samples

        return self._generate_pol(samples)

    def _generate_pol(self, samples: pd.DataFrame) -> pd.DataFrame:
        features = samples.columns
        pol = {}

        for feature in features:
            samples_feature = samples[feature]
            x_mult = pd.Series(np.ones(len(samples_feature)))

            for i in range(self.deg):
                key = '{}^{}'.format(feature, i+1)
                x_mult *= samples_feature
                pol[key] = x_mult.values

        return pd.DataFrame(pol)
