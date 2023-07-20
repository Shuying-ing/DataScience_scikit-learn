"""1.1.11. Logistic regression
    1.1.11.2. Multinomial Case

    Comparison of multinomial logistic L1 vs one-versus-rest L1 logistic regression to classify documents

Lessons learned
-----
- In logistic regression, there are two strategies for multi-class classification: multinomial and one-vs-rest (OvR).
- Multinomial logistic regression is used when the target variable has more than two classes,
    while OvR logistic regression is used when the target variable has two or more classes.
- In OvR logistic regression, a separate binary classifier is trained for each class,
    which distinguishes that class from all other classes.
- In multinomial logistic regression, a single classifier is used to distinguish between all classes at once.
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import timeit
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_dataset(n_samples):
    """Load the 20 newsgroups text dataset"""

    X, y = datasets.fetch_20newsgroups_vectorized(subset="all", return_X_y=True)
    X = X[:n_samples]
    y = y[:n_samples]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, stratify=y, test_size=0.1
    )  # stratify=y-保持测试集中的数据比例和整个数据集的数据比例一致
    n_train_samples, n_features = X_train.shape
    n_classes = np.unique(y).shape[0]
    print(
        "Dataset 20newsgroup, n_train_samples=%i, n_fratures=%i, n_classes=%i"
        % (n_train_samples, n_features, n_classes)
    )  # 对任一数据集，关注的三个数目

    return X_train, X_test, y_train, y_test, n_classes


def model(X_train, X_test, y_train, y_test, n_classes):
    """Model based on multinomial logistic L1 vs one-versus-rest L1 logistic regression"""

    # 考虑两个模型
    models = {
        "ovr": {"name": "One verus Rest", "iter": [1, 2, 3]},
        "multinomial": {"name": "Multinomial", "iter": [1, 2, 5]},
    }

    # 遍历每个模型，ovr或multinomial
    for model in models:
        accuracies = [1 / n_classes]
        times = [0]
        model_params = models[model]

        # 遍历迭代次数
        for this_max_iter in model_params["iter"]:
            # 拟合
            lr = LogisticRegression(
                solver="saga",
                multi_class=model,
                penalty="l1",
                max_iter=this_max_iter,
                random_state=42,
            )  # solver-1.1.11.3. Solvers
            t1 = timeit.default_timer()
            lr.fit(X_train, y_train)
            train_time = timeit.default_timer() - t1
            y_pred = lr.predict(X_test)
            accuracy = np.sum(y_pred == y_test) / y_test.shape[0]  # 统计准确率

            times.append(round(train_time, 4))  # round-保留四位小数
            accuracies.append(accuracy)

        models[model]["times"] = times
        models[model]["accuracies"] = accuracies
        print(pd.DataFrame(models))

    # 可视化——两种模型下不同拟合次数的准确率
    fig = plt.figure()
    for model in models:
        name = models[model]["name"]
        times = models[model]["times"]
        accuracies = models[model]["accuracies"]
        plt.plot(times, accuracies, marker="o", label="Model: %s" % name)
    plt.xlabel("Train time (s)")
    plt.ylabel("Test accuracy")
    plt.legend()
    fig.tight_layout()


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore", category=ConvergenceWarning, module="sklearn"
    )  # filterwarnings-忽略警告

    X_train, X_test, y_train, y_test, n_classes = load_dataset(
        n_samples=5000
    )  # n_samples-对结果影响挺大hhh
    model(X_train, X_test, y_train, y_test, n_classes)

    plt.show()
