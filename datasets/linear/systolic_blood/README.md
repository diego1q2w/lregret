# Systolic Blood Pressure
`Dificulty  level: easy`

Is it possible to find any correlation between age, weight and the systolic blood pressure (SBD)?

That's the question we are looking forward to answer with the following [dataset](http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/frame.html).

Each row has three values `(X1, X2, X3)`.

* `X1` Systolic blood pressure
* `X2` Age in years
* `X3` Weight in pounds

### Usage

```python
lr = LinearRegression()
s = SystolicBlood(lr)
s.fit()
```
### Expected result

It might be slightly different for different values and different iterations
```
--- Result for Gradient Decedent ---
Weights:  
X2      0.224582
X3      0.696740
bias    0.003557
dtype: float64
R-squared:  0.9543378747364191
```

Plus some graphics with the input, the predicted and target SBD and the cost (error) function for each iteration.