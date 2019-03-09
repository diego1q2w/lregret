# Computer Hardware
`Dificulty  level: medium`

Estimating the relative performance (ERP) is something a real geek is up to, so many discussions you might had 
trying to answer the eternal question does the vendor really matter or have more relevance the features?

That's the question we are looking forward to answer with the following [data set](https://archive.ics.uci.edu/ml/datasets/Computer+Hardware).

Each row has 10 values described bellow.

* `vendorName` 30: (adviser, amdahl,apollo, basf, bti, burroughs, c.r.d, cambex, cdc, dec, dg, formation, four-phase, gould, honeywell, hp, ibm, ipl, magnuson, 
microdata, nas, ncr, nixdorf, perkin-elmer, prime, siemens, sperry, sratus, wang)
* `Model` Many unique symbols (string)
* `MYCT` Minimum main memory in kilobytes (integer)
* `MMIN` Minimum main memory in kilobytes (integer)
* `MMAX` Maximum main memory in kilobytes (integer)
* `CACH` Cache memory in kilobytes (integer)
* `CHMIN` Minimum channels in units (integer)
* `CHMAX` Maximum channels in units (integer)
* `PRP` Published relative performance (integer)
* `ERP` Estimated relative performance from the original article (integer)
### Usage

```python
lr = LinearRegression()
s = ComputerHardwareProblem(lr)
s.fit_solving()
```
OR depends how you handle the vendor name but if you choose to use one-hot encoding, you might have more features than 
the ones you really need encouraging over-fitting a state where ML algorithms tend to stay away and when it comes to linear regression
L1 regularisation is perfect to deal with that kind of issues

```python
lr = LinearRegression(learning_rate=0.000000001)
s = ComputerHardwareProblem(lr)
s.fit_l1(0.2)
```
### Expected result

It might differ between values and iterations
```
--- Result for Computer Hardware Data using L1 Regularisation ---
Weights: 
 adviser         1.016475e-04
amdahl          1.345712e-04
apollo         -6.057707e-06
basf           -6.105115e-05
bti            -6.982443e-06
burroughs      -1.056653e-04
c.r.d          -3.874293e-05
cambex         -1.386901e-04
cdc            -2.691313e-04
dec            -7.231359e-05
dg             -1.263898e-04
formation       2.634480e-05
four-phase     -6.733419e-09
gould          -9.066825e-05
harris         -1.033852e-04
honeywell      -2.409332e-04
hp             -7.255046e-05
ibm            -6.252735e-04
ipl            -2.187748e-04
magnuson       -1.178564e-04
microdata      -2.333496e-06
nas            -6.079035e-04
ncr            -1.960323e-04
nixdorf        -9.387040e-06
perkin-elmer   -4.923703e-05
prime          -7.805120e-05
siemens        -3.142153e-04
sperry          5.667771e-04
sratus         -3.147814e-05
wang           -1.685535e-05
MYCT           -1.427511e-02
MMIN            5.619636e-03
MMAX            3.214879e-03
CACH            7.554215e-02
CHMIN           6.622107e-03
CHMAX           6.861718e-02
PRP             5.160429e-01
bias           -2.774955e-03
dtype: float64
R-squared:  0.9415964140073412
----------------------------
```

Plus some graphics with the input, the predicted and target ERM and if applicable the cost (error) function for each iteration.