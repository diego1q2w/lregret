# Data set can be found in:
# https://archive.ics.uci.edu/ml/datasets/Parking+Birmingham
#
# Data Set Information:
#
# Occupancy rates (8:00 to 16:30) from 2016/10/04 to 2016/12/19
#
# Attribute Information:
#
# SystemCodeNumber: Car park ID
# Capacity: Car park capacity
# Occupancy: Car park occupancy rate
# LastUpdated: Date and Time of the measure
from datetime import datetime
import time

import pandas as pd

from datasets.linear import LinearProblem
from regresion.linear.linear import LinearRegression


class ParkingProblem(LinearProblem):

    def dataset_title(self) -> str:
        return "Parking"

    def __init__(self, regression: LinearRegression) -> None:
        df = pd.read_csv('dataset.csv')

        def format_date(raw_date: str) -> float:
            timestamp = time.mktime(datetime.strptime(raw_date, '%Y-%m-%d %H:%M:%S').timetuple())
            delta = timestamp - 1475539200.0
            return delta / 60.0

        df['LastUpdated'] = df['LastUpdated'] \
            .apply(format_date)
        super().__init__(df[['Capacity', 'LastUpdated']], df['Occupancy'], regression)


# lr = LinearRegression()
# p = ParkingProblem(lr)
# p.fit_solving()
