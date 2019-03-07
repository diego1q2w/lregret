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
import matplotlib.pyplot as plt

import pandas as pd
from regresion.linear.linear import LinearRegression

df = pd.read_csv('dataset.csv')
df['LastUpdated'] = df['LastUpdated']\
    .apply(lambda x: time.mktime(datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timetuple()))

Y = df['Occupancy']
X = df[['Capacity', 'LastUpdated']]

lr = LinearRegression(X, Y)
lr.fit()

plt.plot(Y, label='target')
plt.plot(lr.prediction(), label='prediction')
plt.legend()
plt.show()
print("the r-squared is: {}".format(lr.r2()))

