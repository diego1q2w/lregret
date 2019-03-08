# Data set can be found in:
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/frame.html
# Fire and Theft in Chicago
#
# In the following data pairs
# X = fires per 1000 housing units
# Y = thefts per 1000 population
# within the same Zip code in the Chicago metro area
# Reference: U.S. Commission on Civil Rights

import operator

import pandas as pd
import matplotlib.pyplot as plt
from regresion.linear.linear import LinearRegression
from regresion.linear.feature import PolFeatures

df = pd.read_excel('dataset.xls')

Y = df['Y']
X = df['X']

lr = LinearRegression(X, Y)
lr.fit_l2(10000.0)
Yhat = lr.prediction()

sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X, Yhat), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='r')
plt.title("l2")

# Creates a polinomial degree from 1 to 9
for deg in range(6):
    p_features = PolFeatures(deg + 1)
    polinomial = p_features.generate_pol(X)

    lr = LinearRegression(polinomial, Y)
    lr.fit()

    Yhat = lr.prediction()
    plt.scatter(X, Y)
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X, Yhat), key=sort_axis)
    x, y_poly_pred = zip(*sorted_zip)
    plt.plot(x, y_poly_pred, color='r')
    plt.title("deg = {}".format(deg+1))
    plt.show()
    print("the r-squared is: {} for a degree: {}".format(lr.r2(), deg+1))
