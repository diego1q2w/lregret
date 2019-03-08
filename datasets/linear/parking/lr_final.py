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


class ParkingProblem(LinearProblem):

    def dataset_title(self) -> str:
        return "Parking"

    def __init__(self):
        df = pd.read_csv('dataset.csv')
        df['LastUpdated'] = df['LastUpdated'] \
            .apply(lambda x: time.mktime(datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timetuple()))
        super().__init__(df[['Capacity', 'LastUpdated']], df['Occupancy'])


# p = ParkingProblem()
# p.fit_solving()
