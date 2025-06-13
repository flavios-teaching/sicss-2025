# Selecting the best model

## Overfitting and underfitting
https://esciencecenter-digital-skills.github.io/scikit-learn-mooc/overfit/overfitting_vs_under_fitting_slides.html

which data to you prefer?
- important to understand when a model generalizes well on unseen data
- polynomial fct perfectly predicts training set
- why would you prefer the straight line? how do we judge this?
- by looking at test set -> orange points.
- linear curve: similar errors on train and test. polynomial: much larger error on test, no error on train. also, predictions *outside* the range of test data

a harder example
- less obvious!
- model complexity: data generating process. generate X, transform with 9th-degree polynomial, add noise
- we don't know this process! we only see the observations
- -> let's fit models with different polynomial degrees
- poly 9 -> does not match closely the gnerative process (in dashed line)
- reason: too complex for the amount of data. even though it is the "ideal" model (reflecting the generative process), its flexibility captures noise
- **overfitting happens when we have not enough data points relative to the noise**
- **underfitting happens when we have relatively much data compared to the noise** the model is too constrained.





# Exercise: overfitting and underfitting

#### 1: A model that is underfitting:

a) is too complex and thus highly flexible
b) is too constrained and thus limited by its expressivity
c) often makes prediction errors, even on training samples
d) focuses too much on noisy details of the training set

Select all answers that apply
(b), (c)

#### 2: A model that is overfitting:

a) is too complex and thus highly flexible
b) is too constrained and thus limited by its expressivity
c) often makes prediction errors, even on training samples
d) focuses too much on noisy details of the training set

Select all answers that apply

(a), (d)


```python

```


```python

```


```python

```
