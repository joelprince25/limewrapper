LIMEWRAPPER
======================================================================
LimeWrapper provides a simple wrapper around the lime package (https://github.com/marcotcr/lime) to explain the predictions of h2o or sklearn models. While the lime documentation already provides tutorials to do this, the goal of this wrapper is to abstract away the feature engineering processes for end users. This, however, requires model re-training. If that's resource or time intensive, then my recommendation would be directly following the tutorials by the lime authors.

Please refer to the example Jupyter notebook in the example folder to see a classification and regression example

## How to install the package
----------------------------------------------------------------------

To install package dependencies run:

```bash
pip install -r requirements.txt
```
To install the limewrapper package run:

```bash
python setup.py install
```

## Lime Resources
----------------------------------------------------------------------

1. Lime Paper - https://arxiv.org/abs/1602.04938
2. Simple Lime Summary - https://cran.r-project.org/web/packages/lime/vignettes/Understanding_lime.html
3. Lime Tutorials - https://github.com/marcotcr/lime


## How to contribute
----------------------------------------------------------------------
For small bug-fixes and new features, create a separate branch, make necessary commits and create a pull request.
