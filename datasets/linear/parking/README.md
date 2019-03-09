# Parking Birmingham
`Dificulty  level: medium`

Will you be able to predict the occupancy of a parking lot for a given time date?

That's the question we are looking forward to answer with the following [data set](https://archive.ics.uci.edu/ml/datasets/Parking+Birmingham)
with the occupation rates (8:00 to 16:30) from 2016/10/04 to 2016/12/19 for different parking lots in Birmingham, you can start practicing with just one parking lot
then you can use the whole data set to learn from different parking lots and see if you can get a model, how are you going to deal with the date-time
probably timestamp or one-hot encoding?

Each row has four values `(SystemCodeNumber, Capacity, Occupancy, LastUpdated)`.

* `SystemCodeNumber` Car park ID
* `Capacity` Car park capacity
* `Occupancy` Car park occupancy rate
* `LastUpdated`and Time of the measure

### Usage

```python
lr = LinearRegression()
s = ParkingProblem(lr)
s.fit_solving()
```
OR mind you can play with the learning rate to see how that affect in your learning process when using gradient descent
```python
lr = LinearRegression(learning_rate=0.0000001)
s = ParkingProblem(lr)
s.fit()
```
### Expected result

It might differ between values and iterations
```
--- Result for Parking Data using Solving the Weights ---
Weights: 
 [ 4.31965637e-01  1.30586434e-03 -3.42438502e+01]
R-squared:  0.6058529401282189
----------------------------
```

Plus some graphics with the input, the predicted and target occupation and if applicable the cost (error) function for each iteration.