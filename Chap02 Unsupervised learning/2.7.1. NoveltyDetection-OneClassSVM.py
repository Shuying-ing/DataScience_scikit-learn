"""2.7. Novelty and Outlier Detection
    2.7.2. Novelty Detection
    2.7.2.1. Scaling up the One-Class SVM

    > https://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html#sphx-glr-auto-examples-svm-plot-oneclass-py
    
Lessons learned
-----
- One-class SVM , an unsupervised algorithm, learns a decision function for novelty detection
"""

import numpy as np
from sklearn import svm
import matplotlib.font_manager
import matplotlib.pyplot as plt


"""
Generate data
"""
xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 500))
# Ganerate train data
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Ganerate test data - regular novel observations
X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Ganerate test data - abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

"""
Fit the model
"""
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)  # 全是 1
y_pred_test = clf.predict(X_test)  # 全是 1
y_pred_outliers = clf.predict(X_outliers)  # 全是 -1
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

"""
Plot
"""
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])  # (100000, )
Z = Z.reshape(xx.shape)  # (500, 200)

plt.title("Novelty Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="darkred")
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors="palevioletred")

s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="white", s=s, edgecolors="k")
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c="blueviolet", s=s, edgecolors="k")
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c="gold", s=s, edgecolors="k")
plt.axis("tight")
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend(
    [a.collections[0], b1, b2, c],
    [
        "learned frontier",
        "training observations",
        "new regular observations",
        "new abnormal observations",
    ],
    loc="upper left",
    prop=matplotlib.font_manager.FontProperties(size=11),
)
plt.xlabel(
    "error train: %d/200 ; errors novel regular: %d/40 ; errors novel abnormal: %d/40"
    % (n_error_train, n_error_test, n_error_outliers)
)


plt.show()
