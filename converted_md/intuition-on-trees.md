```python

```

# Decision trees
https://esciencecenter-digital-skills.github.io/scikit-learn-mooc/trees/slides.html

- root nodes; decision nodes
- final nodes = leaves - hold prediction values

## Growing trees
classification example
- split 1 -> threshold on Xo
- focus on left/right part. find another optimal decision

regression example
- x is only input variable -> piecewise linear function
- another split etc -> increasingly refined function.
- each split introduces new leaf node
- in each leaf node, we use the average `y` as the prediction

## when to stop? tree depth and overfitting
- black dots - test set; grey dots - train set
- substantial error in first case. "underfitting" -- cannot capture the structure of the data on both the train and test set
- more splits - better performance
- the more leaves, the more we are fitting to single datapoints. the curve perfectly fits the training data, which is both signal and noise
- however, the test error gets larger

## take-home messages
- trees = sequence of simple decision rules
- split data based on threshold
- can repeat splitting on and on
- we don't need to scale numerical features: we consider one feature at a time
- `max_depth` controls the flexibility of the model, or in other words the trade-off between underfitting and overfitting
- mostly useful as building block for ensemble models: random forests, gradient boosting trees. alone, trees are too brittle.



```python

```
