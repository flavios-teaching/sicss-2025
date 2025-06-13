# Advanced topics

https://esciencecenter-digital-skills.github.io/scikit-learn-mooc/advanced_topics/advanced_topics_slides.html


```python
from sklearn.feature_selection import SelectKBest
```


```python
SelectKBest?
```


    [31mInit signature:[39m SelectKBest(score_func=<function f_classif at [32m0x76b577900400[39m>, *, k=[32m10[39m)
    [31mDocstring:[39m     
    Select features according to the k highest scores.
    
    Read more in the :ref:`User Guide <univariate_feature_selection>`.
    
    Parameters
    ----------
    score_func : callable, default=f_classif
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues) or a single array with scores.
        Default is f_classif (see below "See Also"). The default function only
        works with classification tasks.
    
        .. versionadded:: 0.18
    
    k : int or "all", default=10
        Number of top features to select.
        The "all" option bypasses selection, for use in a parameter search.
    
    Attributes
    ----------
    scores_ : array-like of shape (n_features,)
        Scores of features.
    
    pvalues_ : array-like of shape (n_features,)
        p-values of feature scores, None if `score_func` returned only scores.
    
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    
        .. versionadded:: 0.24
    
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    
        .. versionadded:: 1.0
    
    See Also
    --------
    f_classif: ANOVA F-value between label/feature for classification tasks.
    mutual_info_classif: Mutual information for a discrete target.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    f_regression: F-value between label/feature for regression tasks.
    mutual_info_regression: Mutual information for a continuous target.
    SelectPercentile: Select features based on percentile of the highest
        scores.
    SelectFpr : Select features based on a false positive rate test.
    SelectFdr : Select features based on an estimated false discovery rate.
    SelectFwe : Select features based on family-wise error rate.
    GenericUnivariateSelect : Univariate feature selector with configurable
        mode.
    
    Notes
    -----
    Ties between features with equal scores will be broken in an unspecified
    way.
    
    This filter supports unsupervised feature selection that only requests `X` for
    computing the scores.
    
    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.feature_selection import SelectKBest, chi2
    >>> X, y = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> X_new = SelectKBest(chi2, k=20).fit_transform(X, y)
    >>> X_new.shape
    (1797, 20)
    [31mFile:[39m           ~/repositories/teaching/sicss-2025/teaching-notebooks/ml_workshop/lib/python3.12/site-packages/sklearn/feature_selection/_univariate_selection.py
    [31mType:[39m           ABCMeta
    [31mSubclasses:[39m     



```python

```
