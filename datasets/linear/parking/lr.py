import os
from datetime import datetime
import time

import pandas as pd

from datasets.linear import LinearProblem
from regresion.linear.feature import PolFeatures
from regresion.linear.linear import LinearRegression


class ParkingProblem(LinearProblem):

    def dataset_title(self) -> str:
        return "Parking"

    def __init__(self, regression: LinearRegression, pol_features=PolFeatures(1)) -> None:
        file_name = os.path.join(os.path.dirname(__file__), 'dataset.csv')
        df = pd.read_csv(file_name)

        def format_date(raw_date: str) -> float:
            timestamp = time.mktime(datetime.strptime(raw_date, '%Y-%m-%d %H:%M:%S').timetuple())
            delta = timestamp - 1475539200.0
            return delta / 60.0

        df['LastUpdated'] = df['LastUpdated'] \
            .apply(format_date)
        super().__init__(df[['Capacity', 'LastUpdated']], df['Occupancy'], regression, pol_features)


# lr = LinearRegression()
# p = ParkingProblem(lr)
# p.fit_solving()
