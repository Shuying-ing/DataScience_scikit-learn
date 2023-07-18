from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1314)

"""
1.1.1.2. Ordinary Least Squares Complexity
1.1.2.3. Ridge Complexity
1.1.3.1. Setting regularization parameter
    - Lasso model selection: AIC-BIC / cross-validation
    - Lasso model selection via information criteria
"""


def Ordinary_Least_Squares_Linear_Regression():
    """1.1.1. Ordinary Least Squares"""

    # 准备数据
    X = [[0, 0], [1, 1], [2, 2]]
    y = [0, 1, 3]
    # 拟合
    reg = linear_model.LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    # 测试
    print("\n------ Ordinary_Least_Squares_Linear_Regression ------")
    print("Coefficients: ", reg.coef_)
    print("MSE: %.2f" % mean_squared_error(y, y_pred))  # %-格式化输出
    r2_score_ = r2_score(y, y_pred)  # r2_score-决定系数
    print("R2 score: %.2f" % r2_score_)


def NonNegative_Least_Squares_Linear_Regression():
    """1.1.1.1. Non-Negative Least Squares

    Notes
    -----
    适用于非负型数值，例如频率/价格
    NNLS模型回归系数和OLS的高度相关
    """

    # 准备数据
    n_samples, n_features = 200, 50
    X = np.random.randn(n_samples, n_features)  # random.randn-均值为0方差为1的标准正态分布
    true_coef = 3 * np.random.randn(n_features)
    true_coef[true_coef < 0] = 0
    y = np.dot(X, true_coef)
    y += 5 * np.random.normal(size=(n_samples,))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)  # 划分数据集
    # 拟合
    reg_nnls = linear_model.LinearRegression(positive=True)  # 参数positive
    reg_nnls.fit(X_train, y_train)
    y_pred_nnls = reg_nnls.predict(X_test)
    # 测试
    print("\n------ NonNegative_Least_Squares_Linear_Regression ------")
    r2_score_nnls = r2_score(y_test, y_pred_nnls)
    print("R2 score: %.2f" % r2_score_nnls)


def Ridge_Regression(alpha):
    """1.1.2.1. Regression

    Parameters
    ----------
    alpha : float

    Notes
    -----
    使用L2正则化引入惩罚项, 减少模型复杂度避免过拟合
    参数 alpha 越大, shrinkage越强, 系数共线性越强
    """

    # 准备数据
    X = [[0, 0], [1, 1], [2, 2]]
    y = [0, 1, 3]
    # 拟合
    reg = linear_model.Ridge(alpha)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    # 测试
    print("\n------ Ridge_Regression ------")
    print("Coefficients: ", reg.coef_)
    print("Intercepts: ", reg.intercept_)
    print("MSE: %.2f" % mean_squared_error(y, y_pred))
    r2_score_ridge = r2_score(y, y_pred)
    print("R2 score: %.2f" % r2_score_ridge)


def Ridge_Regression_CrossValidation():
    """1.1.2.4. leave-one-out Cross-Validation

    Notes
    -----
    自适应Ridge回归模型中的alpha参数
    """

    # 准备数据
    X = [[0, 0], [1, 1], [2, 2]]
    y = [0, 1, 3]
    # 拟合
    reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))  # logspace-等比数列
    reg.fit(X, y)
    y_pred = reg.predict(X)
    # 测试
    print("\n------ Ridge_Regression_CrossValidation ------")
    print("Coefficients: ", reg.coef_)
    print("Intercepts: ", reg.intercept_)
    print("MSE: %.2f" % mean_squared_error(y, y_pred))
    r2_score_ridgecv = r2_score(y, y_pred)
    print("R2 score: %.2f" % r2_score_ridgecv)


def Lasso(alpha):
    """1.1.3. Lasso

    Parameters
    ----------
    alpha : float

    Notes
    -----
    用于估计sparse coefficients | 能够得到更少的非零系数
    参数 alpha 决定系数稀疏程度
    使用L1正则化引入惩罚项, 减少模型复杂度避免过拟合
    """

    # 准备数据
    X = [[0, 0], [1, 1], [2, 2]]
    y = [0, 1, 3]
    # 拟合
    reg = linear_model.Lasso(alpha)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    # 测试
    print("\n------ Lasso ------")
    r2_score_lasso = r2_score(y, y_pred)
    print("R2 score: %.2f" % r2_score_lasso)


def MultiTask_Lasso(alpha):
    """1.1.4. Multi-task Lasso

    Estimate sparse coefficients for multiple regression problems jointly

    Parameters
    ----------
    alpha : float

    Notes
    -----
    约束条件: 每个任务选择的特征是相同的
    """

    # 准备数据
    rng = np.random.RandomState(42)  # rng-之后得到相同的随机数
    n_samples, n_features, n_tasks = 100, 30, 40
    n_relevant_features = 5
    coef = np.zeros((n_tasks, n_features))
    times = np.linspace(0, 2 * np.pi, n_tasks)  # linspace-等差数列
    for k in range(n_relevant_features):
        coef[:, k] = np.sin((1.0 + rng.randn(1)) * times + 3 * rng.randn(1))
    X = rng.randn(n_samples, n_features)
    y = np.dot(X, coef.T) + rng.randn(n_samples, n_tasks)
    # plt.plot(coef)
    # 拟合
    model_lasso = linear_model.Lasso(alpha)
    model_lasso.fit(X, y)
    model_multi_task_lasso_ = linear_model.MultiTaskLasso(alpha)
    model_multi_task_lasso_.fit(X, y)
    coef_lasso_ = np.array([model_lasso.coef_ for yi in y.T])  # (40, 40, 30)
    coef_lasso_ = np.mean(coef_lasso_, axis=0).reshape(
        (n_tasks, n_features)
    )  # (40, 30), 虽然不知道为什么是按第1列取平均 但的确这样之后的结果是对的...
    coef_multi_task_lasso_ = model_multi_task_lasso_.coef_
    # 可视化
    feature_to_plot = 0
    plt.figure()
    plt.plot(
        coef[:, feature_to_plot], color="seagreen", linewidth=2, label="Ground truth"
    )
    plt.plot(
        coef_lasso_[:, feature_to_plot],
        color="cornflowerblue",
        linewidth=2,
        label="Lasso",
    )
    plt.plot(
        coef_multi_task_lasso_[:, feature_to_plot],
        color="gold",
        linewidth=2,
        label="MultiTaskLasso",
    )
    plt.xlabel("Tasks")
    plt.legend(loc="upper center")
    plt.axis("tight")
    plt.ylim([-1.1, 1.1])


def ElasticNet(alpha, l1_ratio):
    """1.1.5. Elastic-Net

        Examples: L1-based models for Sparse Signals

    Parameters
    ----------
    alpha : float

    l1_ratio : float

    Notes
    -----
    参数 l1_ratio 控制l1和l2的比例
    """

    # 准备数据
    rng = np.random.RandomState(0)
    n_samples, n_features, n_informative = 50, 100, 10
    time_step = np.linspace(-2, 2, n_samples)
    freqs = 2 * np.pi * np.sort(rng.rand(n_features)) / 0.01
    X = np.zeros((n_samples, n_features))
    for i in range(n_features):
        X[:, i] = np.sin(freqs[i] * time_step)  # n_features条sin波形
    idx = np.arange(n_features)
    true_coef = (-1) ** idx * np.exp(-idx / 10)  # 正弦波的y值-泰勒级数近似
    true_coef[n_informative:] = 0  # sparsify coef
    y = np.dot(X, true_coef)
    # sparse, noisy and correlated features
    for i in range(n_features):
        X[:, i] = np.sin(freqs[i] * time_step + 2 * (rng.random_sample() - 0.5))
        X[:, i] += 0.2 * rng.normal(0, 1, n_samples)
    y += 0.2 * rng.normal(0, 1, n_samples)
    # 数据可视化
    plt.plot(time_step, y)
    plt.ylabel("target signal")
    plt.xlabel("time")
    _ = plt.title("Superposition of sinusoidal signals")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, shuffle=False
    )

    # 拟合
    enet = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    enet.fit(X_train, y_train)
    y_pred_enet = enet.predict(X_test)

    # 测试
    print("\n------ ElasticNet ------")
    r2_score_enet = r2_score(y_test, y_pred_enet)
    print("R2 score: %.2f" % r2_score_enet)


if __name__ == "__main__":
    # Ordinary_Least_Squares_Linear_Regression()  # OLS
    # NonNegative_Least_Squares_Linear_Regression()  # NNLS
    # Ridge_Regression(alpha=0.5)  # Ridge
    # Ridge_Regression_CrossValidation()  # RidgeCV
    # Lasso(alpha=0.5)  # Lasso
    # MultiTask_Lasso(alpha=0.5)  # multi-task Lasso
    # ElasticNet(alpha=0.08, l1_ratio=0.5)  # ElasticNet
    plt.show()
