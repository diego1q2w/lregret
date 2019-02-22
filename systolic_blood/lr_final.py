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
from algorithm import LinearRegression

df = pd.read_excel('dataset.xls')

Y = df['X1']
X = df[['X2', 'X3']]

plt.scatter(X['X2'], Y)
plt.scatter(X['X3'], Y)
plt.show()

lr = LinearRegression(X, Y)
lr.fit()
Yhat = lr.prediction()

plt.plot(Y, label='r')
plt.plot(lr.prediction(), label='r')
plt.title("l2")
plt.show()
print("the r-squared is: {}".format(lr.r2()))

