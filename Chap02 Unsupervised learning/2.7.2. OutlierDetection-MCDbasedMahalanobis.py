"""2.7. Novelty and Outlier Detection
    2.7.3. Outlier Detection
    2.7.3.1. Fitting an elliptic envelope

    > https://scikit-learn.org/stable/auto_examples/covariance/plot_mahalanobis_distances.html#sphx-glr-auto-examples-covariance-plot-mahalanobis-distances-py

Lessons learned
-----
For two distributions, one drawn from a contaminating distribution, one coming from the real, Gaussian distribution
- using MCD-based Mahalanobis distances can distinguish them
- but using standard covariance MLE based Mahalanobis distances cannot

Minimum Covariance Determinant estimator (MCD)
"""


import numpy as np
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import matplotlib.pyplot as plt

# for consistent results
np.random.seed(7)

"""
Generate data
"""
n_samples = 125
n_outliers = 25
n_features = 2

# generate Gaussian data of shape (125, 2)
# Both features are Gaussian distributed with mean of 0
# feature 1 has a standard deviation equal to 2
# feature 2 has a standard deviation equal to 1
gen_cov = np.eye(n_features)
gen_cov[0, 0] = 2.0
X = np.dot(np.random.randn(n_samples, n_features), gen_cov)
# add some outliers
# 25 samples are replaced with Gaussian outlier samples
# feature 1 has a standard deviation equal to 1
# feature 2 has a standard deviation equal to 7
outliers_cov = np.eye(n_features)
outliers_cov[np.arange(1, n_features), np.arange(1, n_features)] = 7.0
X[-n_outliers:] = np.dot(np.random.randn(n_outliers, n_features), outliers_cov)

"""
Fit the model
"""
# fit a MCD robust estimator
robust_cov = MinCovDet().fit(X)
# fit a MLE estimator
emp_cov = EmpiricalCovariance().fit(X)
print(
    "Estimated covariance matrix:\nMCD (Robust):\n{}\nMLE:\n{}".format(
        robust_cov.covariance_, emp_cov.covariance_
    )
)

"""
Plot
"""
fig, ax = plt.subplots(figsize=(10, 5))
# Plot data set
inlier_plot = ax.scatter(X[:, 0], X[:, 1], color="black", label="inliers")
outlier_plot = ax.scatter(
    X[:, 0][-n_outliers:], X[:, 1][-n_outliers:], color="red", label="outliers"
)
ax.set_xlim(ax.get_xlim()[0], 10.0)
ax.set_title("Mahalanobis distances of a contaminated data set")

# Create meshgrid of feature 1 and feature 2 values
xx, yy = np.meshgrid(
    np.linspace(plt.xlim()[0], plt.xlim()[1], 100),
    np.linspace(plt.ylim()[0], plt.ylim()[1], 100),
)
zz = np.c_[xx.ravel(), yy.ravel()]

# Calculate the MLE based Mahalanobis distances
mahal_emp_cov = emp_cov.mahalanobis(zz)
mahal_emp_cov = mahal_emp_cov.reshape(xx.shape)
emp_cov_contour = plt.contour(
    xx, yy, np.sqrt(mahal_emp_cov), cmap=plt.cm.PuBu_r, linestyles="dashed"
)
# Calculate the MCD based Mahalanobis distances
mahal_robust_cov = robust_cov.mahalanobis(zz)
mahal_robust_cov = mahal_robust_cov.reshape(xx.shape)
robust_contour = ax.contour(
    xx, yy, np.sqrt(mahal_robust_cov), cmap=plt.cm.YlOrBr_r, linestyles="dotted"
)

ax.legend(
    [
        emp_cov_contour.collections[1],
        robust_contour.collections[1],
        inlier_plot,
        outlier_plot,
    ],
    ["MLE dist", "MCD dist", "inliers", "outliers"],
    loc="upper right",
    borderaxespad=0,
)

plt.show()
