# National Unemployment Male Vs. Female
`Dificulty  level: easy`

Is women's unemployment related to the men's?

That's the question we are looking forward to answer with the following [data set](http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/frame.html).

Each row has two values `(X, Y)`.

* `X` National unemployment rate for adult males
* `Y` National unemployment rate for adult females

### Usage

```python
lr = LinearRegression()
s = UnemploymentProblem(lr)
s.fit_solving()
```
OR you can add feature degrees and see how you can adapt it with the target
```python
lr = LinearRegression()
s = UnemploymentProblem(lr)
p_features = PolFeatures(deg=4)
s.fit_polynomial(p_features)
```
### Expected result

It might differ between values and iterations
```
--- Result for National Unemployment Data using Solving the Weights ---
Weights: 
 [-8.60285716e+00  2.01800680e+00 -1.77098408e-01  5.32430765e-03
  1.59546993e+01]
R-squared:  0.8891322763841704
----------------------------
```

Plus some graphics with the input, the predicted and target unemployment and if applicable the cost (error) function for each iteration.