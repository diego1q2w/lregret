# lRegret

For people who are just starting their way into the world of deep learning by reading a book or probably taking an online course or in school, there are two concepts that are the base for building the foundation of machine learning those are linear regression and logistic regression.

Linear and logistic regression are the must-understand algorithms when it comes to deep learning since both prediction and classification are built on top of them, the idea of a neuron comes from them.

This repository has a different set of datasets with different sizes and a different number of features grouped by application while some of them are more suitable for prediction some others are more for classification, as well as some of them are more suitable for some level of regularisation (L1 and L2) while others not, and that's for you to decide.

The code base already implements the most recommended methods for each dataset but you are free to play as you pleased.

## Data sets

More data sets will be added soon

### Linear Regression

For more details you can go to `datasets/lnear` and select any data-set for more information within the ReadMe

* `Systolic Blood Pressure` for more information go [here](https://github.com/diego1q2w/lregret/tree/master/datasets/linear/systolic_blood#systolic-blood-pressure)
* `Parking Birmingham` for more information go [here](https://github.com/diego1q2w/lregret/tree/master/datasets/linear/parking#parking-birmingham)
* `National Unemployment Male Vs. Female` for more information go [here](https://github.com/diego1q2w/lregret/tree/master/datasets/linear/national_unemployment#national-unemployment-male-vs-female)
* `Fire and Theft in Chicago` for more information go [here](https://github.com/diego1q2w/lregret/tree/master/datasets/linear/fire_and_theft#fire-and-theft-in-chicago)
* `Computer Hardware` for more information go [here](https://github.com/diego1q2w/lregret/tree/master/datasets/linear/computer_hadware#computer-hardware)

### Logistic Regression
 TODO: add data-sets

## Operations

Each operation impact the training in different ways by adding more features or using some kind of regularisation

### Linear Regression
* `fit` The traditional training using gradient descend you might want to set `learning_rate` parameter to adjust it to the data-set
* `fit_l2` Similar to `fit` but adding L2 regularisation a common technique to avoid over-fitting by reducing the weights to values close to zero, don't forget to set the `l2` constant
* `fit_l1` Similar to `fit` but adding L1 regularisation a common technique to avoid over-fitting by muting non-relevant weights, don't forget to set the `l1` constant
* `fit_solving` Linear Regression is one of the few or perhaps the only algorithm that has a solution so it's possible to solve the weights using only the input and the target instead of using gradient descend

For any operation you can use Polinomial Regression, there are different ways to implement it by using the api with the parameter `--degree` (more details [here](https://github.com/diego1q2w/lregret#3--use-a-handy-script)) or call it directly with the class `PolFeatures` 
see an example [here](https://github.com/diego1q2w/lregret/tree/master/datasets/linear/national_unemployment#usage), for more information please check the `How to use it` section.
### Logistic Regression
TODO: implement algorithms

## How to use it 

The context of each user might be different this is why the way of running might vary between computers, here are some ways you can get to play with this project.

### 1- Simply running the file

This is the easiest if you already have a python env set and some experience just go to the each data-set (`datasets/linear`) or (`datasets/logistic`) and run `python lr.py`
play with the input and the constants so you get different results.

Just a small reminder for this option, please make sure you have the packages required by `requirements.txt` installed

### 2- Run the main.py

Going from dataset to dataset might be a little bit overwhelming, surely a small API will make your life easier, in order to get there just run the main file.

For starting you can type `python main.py -h` to get some help about which commands you have available but anyway here you have some examples:

* `python main.py systolic_blood fit_l2 --l2 100`
* `python main.py systolic_blood fit`
* `python main.py systolic_blood fit --degree 9`

Just a small reminder for this option, please make sure you have the packages required by `requirements.txt` installed

### 3- Use a handy script

You are a modern developer and you just use `docker-compose` for everything don't worry got you covered you have a script with the same api described in the above point.

For start you can just run `./script/run -h` to get some help about which commands you have available but anyway here you have some examples:

* `./script/run systolic_blood fit_l2 --l2 100`
* `./script/run systolic_blood fit`
* `./script/run systolic_blood fit --degree 9`

Just a small note about this option, since getting UI features work within docker containers might have a huge impact in the docker image size the graph results are stored in the folder `tmp_figures`,
each time you run this script previous images generated will be deleted.

And don't forget to have installed docker-compose.

The first time you run this script might be a bit slow since it has to fetch the image, don't worry it will be only the first time.

### 4- Use a handy-(ish) script

Did you have issues with the previous script? that provably has to with the volumes created by docker, although it's really rare, it can still happen,
in order to cope with that you can run the script `./script/build_run` with the exact same API described above, so yes the `./script/build_run -h` works here as well.

A small note, since the images were shared through docker-volumes and guess what (this option has the volumes deactivated) you won't be able to see any graph, but you will see the rest fo the output such as trained weights, errors, and r2 factor.
You can override the methods `plot_dataset` and `print_result` in any data-set to do whatever you'd like with the data.

The first time you run this script might be a bit slow since it has to fetch the image, don't worry it will be only the first time.

# What to expect?

A set of handful graphics for most of the options (check the `How to use it` section for more details), and some printed result, remeber you can override in each dataset the methods `plot_dataset` and `print_result` to generate your custom graphics and output.

# Wanna help? 

You create a PR and I'll be happy to get some help :)


### TODOs

- [x] Linear Regression Algorithm
- [x] Gradient Descendent Algorithm
- [x] Linear Regression datasets implementation
- [x] Linear Regression L2 regularisation implementation
- [x] Linear Regression L1 regularisation implementation
- [x] Python API
- [x] Docker implementation
- [ ] Logistic Regression Algorithms
- [ ] Logistic Regression L2 regularisation
- [ ] Logistic Regression L1 regularisation
- [ ] Logistic Regression datasets implementation
- [ ] Add external libraries for comparision
