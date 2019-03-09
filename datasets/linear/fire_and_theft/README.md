# Fire and Theft in Chicago
`Dificulty  level: easy`

Are the fires and theft related?

That's the question we are looking forward to answer with the following [data set](http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/frame.html).

Each row has two values `(X, Y)`.

* `X` Fires per 1000 housing units
* `Y` Thefts per 1000 population

Within the same Zip code in the Chicago metro area
### Usage

```python
lr = LinearRegression()
s = FireAndTheftProblem(lr)
s.fit_solving()
```
OR you can add feature degrees and see how you can adapt it with the target
```python
lr = LinearRegression()
s = FireAndTheftProblem(lr)
p_features = PolFeatures(deg=4)
s.fit_polynomial(p_features)
```
### Expected result

It might differ between values and iterations
```
--- Result for Fire and Theft in Chicago Data using Solving the Weights ---
Weights: 
 [ 7.94256975e+00 -3.38486911e-01 -3.09236469e-03  2.23097657e-04
 -5.71527806e+00]
R-squared:  0.7675731689924846
----------------------------
```

Plus some graphics with the input, the predicted and target thefts and if applicable the cost (error) function for each iteration.