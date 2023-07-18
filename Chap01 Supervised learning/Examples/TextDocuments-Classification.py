"""1.1.2.2. Classification

    Classification of text documents using sparse features

Lessons learned
-----
- Ridge可以作为分类器模型
"""

from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

categories = [
    "alt.atheism",
    "talk.religion.misc",
    "comp.graphics",
    "sci.space",
]


def size_mb(docs):
    """Compute the size(mb) of docs"""

    return sum(len(s.encode("utf-8")) for s in docs) / 1e6


def load_dataset(verbose=False, remove=()):
    """Load and vectorize the 20 newsgroups text dataset"""

    data_train = datasets.fetch_20newsgroups(
        subset="train",
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=remove,
    )  # remove-移除部分metadata

    data_test = datasets.fetch_20newsgroups(
        subset="test",
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=remove,
    )
    # print(data_test.DESCR)  # 查看信息
    # print(data_test.data[:5])
    # print(data_test.target[:5])   # data和target分开查看

    y_train, y_test = data_train.target, data_test.target

    # 特征提取 TF-IDF
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
    )  # 文本数据-->TermFrequency-InverseDocumentFrequency特征向量，sublinear_tf-子线性缩放
    X_train = vectorizer.fit_transform(data_train.data)  # 先fit-拟合数据学习规律 再transform-标准化
    X_test = vectorizer.transform(data_test.data)  # transform-使用data_train中学到的规律进行标准化

    feature_names = vectorizer.get_feature_names_out()
    target_names = data_train.target_names  # 可能不等于`categories`

    if verbose:  # 输出详细信息
        data_train_size_mb = size_mb(data_train.data)
        data_test_size_mb = size_mb(data_test.data)
        print(
            f"{len(data_train.data)} documents - "
            f"{data_train_size_mb:.2f}MB (training set)"
        )  # f-格式化输出
        print(
            f"{len(data_test.data)} documents - "
            f"{data_test_size_mb:.2f}MB (testing set)"
        )
        print(f"{len(target_names)} categories")
        print(
            f"(training set) n_samples: {X_train.shape[0]}, n_features: {X_train.shape[1]}"
        )  # X_train维度=样本数×特征数
        print(
            f"(testing set) n_samples: {X_test.shape[0]}, n_features: {X_test.shape[1]}"
        )

    return X_train, X_test, y_train, y_test, feature_names, target_names


def model(X_train, X_test, y_train, y_test, feature_names, target_names):
    """Model based on RidgeClassifier

    Notes
    -----
    使用LogisticRegression和RidgeClassifier 准确率应当差不多
    但RidgeClassifieresp计算效率更高 (只计算一次投影矩阵), esp对类别较多的数据
    """

    # 拟合
    clf = linear_model.RidgeClassifier(tol=1e-2, solver="sparse_cg")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # 测试
    fig, ax = plt.subplots(figsize=(10, 5))
    # ConfusionMatrixDisplay.from_estimator(estimator=clf, X=X_test, y=y_test)  # 混淆矩阵
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)  # 这两行结果应该是一样的
    ax.xaxis.set_ticklabels(target_names)
    ax.yaxis.set_ticklabels(target_names)
    ax.set_title(
        f"Confusion Matrix for {clf.__class__.__name__}\non the original documents"
    )

    return clf


def plot_feature_effects(clf, target_names):
    """Plot the most influenciing features of each target

    Notes
    -----
    快学学别人的条形图怎么画的!
    """

    # 计算每个特征对于每个类别的平均特征影响
    average_feature_effects = (
        clf.coef_ * np.asarray(X_train.mean(axis=0)).ravel()
    )  # ravel-将多维数组转化为一维数组
    # print(average_feature_effects.shape)  # 维度 = len(target)×len(features)

    # 针对每个类别，获取最影响它的5个特征
    for i, label in enumerate(target_names):
        top5 = np.argsort(average_feature_effects[i])[-5:][
            ::-1
        ]  # argsort-[-5:]返回数组中值最大的前5个元素的索引, [::-1]将索引反转
        if i == 0:
            top = pd.DataFrame(feature_names[top5], columns=[label])
            top_indices = top5
        else:
            top[label] = feature_names[top5]
            top_indices = np.concatenate(
                (top_indices, top5), axis=None
            )  # concatenate-数组拼接
    top_indices = np.unique(top_indices)
    predictive_words = feature_names[top_indices]
    print("Top 5 keywords per class:\n", top)

    # 可视化——条形图
    bar_size = 0.25  # 单个条形的高度
    padding = 0.75  # 条形之间的间距
    y_locs = np.arange(len(top_indices)) * (4 * bar_size + padding)  # 每个条形的y坐标

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, label in enumerate(target_names):
        ax.barh(
            y_locs + (i - 2) * bar_size,
            average_feature_effects[i, top_indices],
            height=bar_size,
            label=label,
        )  # barh-条形图
    ax.set(
        yticks=y_locs,
        yticklabels=predictive_words,
        ylim=[
            0 - 4 * bar_size,
            len(top_indices) * (4 * bar_size + padding) - 4 * bar_size,
        ],
    )
    ax.legend(loc="lower right")
    ax.set_title("Average feature effect on the original data")


if __name__ == "__main__":
    # without metadata stripping
    X_train, X_test, y_train, y_test, feature_names, target_names = load_dataset(
        verbose=True, remove=()
    )
    clf = model(X_train, X_test, y_train, y_test, feature_names, target_names)
    plot_feature_effects(clf, target_names)

    # with metadata stripping
    X_train, X_test, y_train, y_test, feature_names, target_names = load_dataset(
        verbose=True, remove=("headers", "footers", "quotes")
    )
    clf = model(X_train, X_test, y_train, y_test, feature_names, target_names)
    plot_feature_effects(clf, target_names)

    plt.show()
