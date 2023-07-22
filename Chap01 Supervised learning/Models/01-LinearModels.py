"""
Maybe Useful
--------
1.1.1.2. Ordinary Least Squares Complexity
1.1.2.3. Ridge Complexity
1.1.3.1. Setting regularization parameter
    - Lasso model selection: AIC-BIC / cross-validation
    - Lasso model selection via information criteria
1.1.6. Multi-task Elastic-Net
1.1.7. Least Angle Regression
"""

from sklearn import linear_model, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import seaborn as sns
import numpy as np
import pandas as pd

np.random.seed(1314)


def Ordinary_LeastSquares_LinearRegression():
    """1.1.1. Ordinary Least Squares

    Fit a linear model with coefficients to minimize the residual sum of squares between the observed targets

    Notes
    -----
    the coefficient estimates rely on the independence of the features
    """

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


def NonNegative_LeastSquares_LinearRegression():
    """1.1.1.1. Non-Negative Least Squares

    Fit a linear model with non-negative coefficients

    Notes
    -----
    be useful to represent some physical or naturally non-negative quantities
        (e.g., frequency counts or prices of goods)
    the regression coefficients between OLS and NNLS are highly correlated
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

    Estimate coefficients with l2-norm regularization of the coefficients

    Parameters
    ----------
    alpha : float

    Notes
    -----
    introduce l_2 regularization term to impose a penalty on the size of coefficients
    addresse some of the problems of OLS
    the larger the value of alpha,
        the greater the amount of shrinkage,
            the coefficients become more robust to collinearity
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

    Implement ridge regression with built-in cross-validation of the alpha parameter
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


def Lasso_Regression(alpha):
    """1.1.3. Lasso

    Estimate sparse coefficients with l1-norm regularization of the coefficients

    Parameters
    ----------
    alpha : float

    Notes
    -----
    introduce l_1 regularization term to impose a penalty on the size of coefficients
    bsed on coordinate descent
    prefer solutions with fewer non-zero coefficients
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


def MultiTask_Lasso_Regression(alpha):
    """1.1.4. Multi-task Lasso

    Estimate sparse coefficients for multiple regression problems jointly

    Parameters
    ----------
    alpha : float

    Notes
    -----
    Constraint: selected features are the same for all regression problems/tasks
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

    Estimate coefficients with both l1 and l2-norm regularization of the coefficients

    Parameters
    ----------
    alpha : float

    l1_ratio : float

    Notes
    -----
    be useful to deal with multiple correlated features
    l1_ratio controls the strength of l_1 regularization vs. l_2 regularization
    """

    # 准备数据 - Sparse Signals
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


def LARSLasso_Regression(alpha):
    """1.1.8. LARS Lasso

    Estimate sparse coefficients with lasso model using the LARS algorithm

    Parameters
    ----------
    alpha : float

    Notes
    -----
    do not based on coordinate descent
    yield exact solution, which is piecewise linear as a function of the norm of its coefficients
    """

    # 准备数据
    X = [[0, 0], [1, 1], [2, 2]]
    y = [0, 1, 3]
    # 拟合
    reg = linear_model.LassoLars(alpha)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    # 测试
    print("\n------ LARSLasso ------")
    r2_score_LARSLasso = r2_score(y, y_pred)
    print("R2 score: %.2f" % r2_score_LARSLasso)


def Orthogonal_Matching_Pursuit():
    """1.1.9. Orthogonal Matching Pursuit (OMP)

    Approximate the fit of a linear model with constraints imposed on the number of non-zero coefficients

    Notes
    -----
    based on a greedy algorithm
    a forward feature selection method
    be used to recover a sparse signal
    """

    # 准备数据 - 稀疏信号
    n_components, n_features = 512, 100
    n_nonzero_coefs = 17
    y, X, w = datasets.make_sparse_coded_signal(
        n_samples=1,
        n_components=n_components,
        n_features=n_features,
        n_nonzero_coefs=n_nonzero_coefs,
        random_state=0,
    )  # w-每个样本的稀疏编码系数
    # print(w.shape)  # (512,)
    y_noisy = y + 0.05 * np.random.randn(len(y))
    (idx,) = w.nonzero()  # nonzero-返回数组w中非零元素的下标
    plt.figure(figsize=(7, 7))
    plt.subplot(4, 1, 1)
    plt.stem(idx, w[idx])  # stem-离散线型图！！
    plt.xlim(0, 512)
    plt.title("Sparse signal")

    # 拟合原数据 + 可视化
    omp = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    omp.fit(X, y)
    coef = omp.coef_
    (idx_r,) = coef.nonzero()
    plt.subplot(4, 1, 2)
    plt.stem(idx_r, coef[idx_r])
    plt.xlim(0, 512)
    plt.title("Recovered signal from noise-free measurements")

    # 拟合加噪数据 + 可视化
    omp.fit(X, y_noisy)
    coef = omp.coef_
    (idx_r,) = coef.nonzero()
    plt.subplot(4, 1, 3)
    plt.stem(idx_r, coef[idx_r])
    plt.xlim(0, 512)
    plt.title("Recovered signal from noisy measurements")

    # 基于CV拟合加噪数据 + 可视化
    omp_cv = linear_model.OrthogonalMatchingPursuitCV()
    omp_cv.fit(X, y_noisy)
    coef = omp_cv.coef_
    (idx_r,) = coef.nonzero()
    plt.subplot(4, 1, 4)
    plt.stem(idx_r, coef[idx_r])
    plt.xlim(0, 512)
    plt.title("Recovered signal from noisy measurements with CV")
    plt.tight_layout()


def BayesianRidge_Regression():
    """1.1.10.1. Bayesian Ridge Regression

    Estimate a probabilistic model of the regression problem
        with regularization parameters included in the estimation procedure

    Notes
    -----
    Bayesian regression: the regularization parameter is not set in a hard sense but tuned to the data at hand
    the selection of initial values of the regularization parameters (alpha, lambda) may be important
    the prior for the coefficient is given by a spherical Gaussian
    the regularization parameters alpha/lambda and being estimated by maximizing the log marginal likelihood
    """

    # 准备数据
    X = [[0, 0], [1, 1], [2, 2]]
    y = [0, 1, 3]
    # 拟合
    reg = linear_model.BayesianRidge()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    # 测试
    print("\n------ BayesianRidge_Regression ------")
    print("Coefficients: ", reg.coef_)
    print("Intercepts: ", reg.intercept_)
    print("MSE: %.2f" % mean_squared_error(y, y_pred))
    r2_score_ridge = r2_score(y, y_pred)
    print("R2 score: %.2f" % r2_score_ridge)


def Automatic_Relevance_Determination():
    """1.1.10.2. Automatic Relevance Determination - ARD

    Estimate a probabilistic model of the regression problem similar to BayesianRidge_Regression,
        but that leads to sparser coefficients

    Notes
    -----
    the prior for the coefficient drops the spherical Gaussian distribution for a centered elliptic Gaussian distribution
        --> each coefficient can itself be drawn from a Gaussian distribution
    ARD = Sparse Bayesian Learning and Relevance Vector Machine
    """

    # 准备数据
    X, y, true_weights = datasets.make_regression(
        n_samples=100,
        n_features=100,
        n_informative=10,
        noise=8,
        coef=True,
        random_state=42,
    )  # informative_feature-帮助 predict the target variable 的特征
    # 拟合
    olr = linear_model.LinearRegression().fit(X, y)
    brr = linear_model.BayesianRidge().fit(X, y)
    ard = linear_model.ARDRegression().fit(X, y)
    df = pd.DataFrame(
        {
            "Weights of true generative process": true_weights,
            "ARDRegression": ard.coef_,
            "BayesianRidge": brr.coef_,
            "LinearRegression": olr.coef_,
        }
    )  # (100, 4)
    # 可视化——每种模型下的系数
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        df.T,
        norm=SymLogNorm(linthresh=10e-4, vmin=-80, vmax=80),
        cbar_kws={"label": "coefficients' values"},
        cmap="seismic_r",
    )  # SymLogNorm-scale数据
    plt.ylabel("linear model")
    plt.xlabel("cofficients")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    _ = plt.title("Models' cofficients")


def Logistic_Regression(C, penalty):
    """1.1.11. Logistic regression

    Estimate the probability of an event occurring

    Parameters
    ----------
    C : float

    penalty : {'l1', 'l2', 'elasticnet', 'none'}

    Notes
    -----
    logistic regression = logit regression
        = maximum-entropy classification (MaxEnt)
        = log-linear classifier
    parameter C is inverse of regularization strength
    be implemented as a linear model for classification rather than regression in terms of the scikit-learn/ML nomenclature
    a special case of the Generalized Linear Models (GLM)
    have many many solvers, see 1.1.11.3. Solvers
    """

    # 准备数据
    X, y = datasets.load_digits(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    y = (y > 4).astype(int)  # astype-把True值置为1 False值置为0

    fig, axes = plt.subplots(3, 3)
    for i, (C, axes_row) in enumerate(zip((1, 0.1, 0.01), axes)):
        # 拟合——不同的C/不同的正则化项
        clf_l1_LR = linear_model.LogisticRegression(
            C=C, penalty="l1", tol=0.01, solver="saga"
        )
        clf_l2_LR = linear_model.LogisticRegression(
            C=C, penalty="l2", tol=0.01, solver="saga"
        )
        clf_en_LR = linear_model.LogisticRegression(
            C=C, penalty="elasticnet", solver="saga", l1_ratio=0.05, tol=0.01
        )
        clf_l1_LR.fit(X, y)
        clf_l2_LR.fit(X, y)
        clf_en_LR.fit(X, y)

        coef_l1_LR = clf_l1_LR.coef_.ravel()
        coef_l2_LR = clf_l2_LR.coef_.ravel()
        coef_en_LR = clf_en_LR.coef_.ravel()

        # 可视化
        if i == 0:
            axes_row[0].set_title("L1 penalty")
            axes_row[1].set_title("Elastic-Net\nl1_ratio = %s" % 0.05)
            axes_row[2].set_title("L2 penalty")

        for ax, coefs in zip(axes_row, [coef_l1_LR, coef_en_LR, coef_l2_LR]):
            ax.imshow(
                np.abs(coefs.reshape(8, 8)),
                interpolation="nearest",
                cmap="binary",
                vmax=1,
                vmin=0,
            )
            ax.set_xticks(())
            ax.set_yticks(())
        axes_row[0].set_ylabel("C = %s" % C)
        plt.tight_layout()


def GeneralizedLinearModels():
    """1.1.12. Generalized Linear Models

    An extension of linear models

    Notes
    -----
    GLM extend linear models:
        1. the predicted values are linked to a linear combination of the input variables
        2. the squared loss function is replaced by the unit deviance of a distribution
    The choice of the distribution depends on the problem at hand.
    """

    # 准备数据
    X = [[0, 0], [1, 1], [2, 2]]
    y = [0, 1, 3]
    # 拟合
    reg = linear_model.TweedieRegressor(
        power=1, alpha=0.5, link="log"
    )  # power-拟合分布，详见 1.1.12.1. Usage
    reg.fit(X, y)
    y_pred = reg.predict(X)
    # 测试
    print("\n------ GeneralizedLinearModels ------")
    r2_score_lasso = r2_score(y, y_pred)
    print("R2 score: %.2f" % r2_score_lasso)


def RANdom_SAmple_Consensus():
    """1.1.16.2. RANSAC: RANdom SAmple Consensus

    Fit a regression model in the presence of corrupt data: either outliers,
        or error in the model.

    Notes
    -----
    fits a model from random subsets of inliers from the complete data set
    a non-deterministic algorithm producing only a reasonable result with a certain probability
    be typically used for linear and non-linear regression problems
    be especially popular in the field of photogrammetric computer vision
    """

    # 准备数据
    n_samples = 1000
    n_outliers = 50
    X, y, coef = datasets.make_regression(
        n_samples=n_samples,
        n_features=1,
        n_informative=1,
        noise=10,
        coef=True,
        random_state=0,
    )
    # Add outlier data
    np.random.seed(0)
    X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
    y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

    # 拟合——普通模型（所有数据）
    lr = linear_model.LinearRegression()
    lr.fit(X, y)
    # 拟合——ransac模型（抽取数据）
    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_  # 找出非异常点
    outlier_mask = np.logical_not(inlier_mask)  # 取反

    # 测试
    line_X = np.arange(X.min(), X.max())[:, np.newaxis]  # newaxis-升维
    line_y = lr.predict(line_X)
    line_y_ransac = ransac.predict(line_X)
    print("Estimated coefficients (true, linear regression, RANSAC):")
    print(coef, lr.coef_, ransac.estimator_.coef_)
    plt.scatter(
        X[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
    )
    plt.scatter(
        X[outlier_mask], y[outlier_mask], color="gold", marker=".", label="Outliers"
    )
    plt.plot(line_X, line_y, color="navy", linewidth=2, label="Linear regressor")
    plt.plot(
        line_X,
        line_y_ransac,
        color="cornflowerblue",
        linewidth=2,
        label="RANSAC regressor",
    )
    plt.legend(loc="lower right")
    plt.xlabel("Input")
    plt.ylabel("Response")


def TheilSen_Regression():
    """1.1.16.3. Theil-Sen estimator: generalized-median-based estimator

    Fit a regression model in the presence of multivariate outliers

    Notes
    -----
    uses a generalization of the median in multiple dimensions
    be robust to multivariate outliers
    loses its robustness properties and becomes no better than an ordinary least squares in high dimension
    In certain cases Theil-Sen performs better than RANSAC which is also a robust method.
    """

    # 准备数据
    np.random.seed(0)
    n_samples = 200
    X = np.random.randn(n_samples)
    w = 3.0
    c = 2.0
    noise = 0.1 * np.random.randn(n_samples)
    y = w * X + c + noise
    X1 = X
    y1 = y

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    estimators = [
        ("OLS", linear_model.LinearRegression()),
        ("Theil-Sen", linear_model.TheilSenRegressor(random_state=42)),
        ("RANSAC", linear_model.RANSACRegressor(random_state=42)),
    ]
    colors = {"OLS": "turquoise", "Theil-Sen": "gold", "RANSAC": "lightgreen"}

    # 拟合 + 可视化
    # 10% outliers [Outliers only in the y direction]
    y[-20:] += -20 * X[-20:]
    X = X[:, np.newaxis]
    line_x = np.array([-3, 3])  # (2,)
    axs[0].scatter(X, y, color="indigo", marker="x", s=40)
    for name, estimator in estimators:
        estimator.fit(X, y)
        y_pred = estimator.predict(line_x.reshape(2, 1))
        axs[0].plot(
            line_x,
            y_pred,
            color=colors[name],
            linewidth=2,
            label="%s" % (name),
        )
    axs[0].legend(loc="upper left")
    axs[0].set_title("Corrupt y")

    # 10% outliers [Outliers only in the x direction]
    X1[-20:] = 9.9
    y1[-20:] += 22
    X1 = X1[:, np.newaxis]
    line_x = np.array([-3, 10])  # (2,)
    axs[1].scatter(X1, y1, color="indigo", marker="x", s=40)
    for name, estimator in estimators:
        estimator.fit(X1, y1)
        y_pred = estimator.predict(line_x.reshape(2, 1))
        axs[1].plot(
            line_x,
            y_pred,
            color=colors[name],
            linewidth=2,
            label="%s" % (name),
        )
    axs[1].legend(loc="upper left")
    axs[1].set_title("Corrupt x")


def Quantile_Regressor():
    """1.1.17. Quantile Regression

    Estimate the median or other quantiles of y conditional on X

    Notes
    -----
    As the pinball loss is only linear in the residuals,
        quantile regression is much more robust to outliers than squared error based estimation of the mean.
    may be useful in predicting an interval instead of point prediction
    """

    # 准备数据
    rng = np.random.RandomState(42)
    x = np.linspace(start=0, stop=10, num=100)
    X = x[:, np.newaxis]
    y_true_mean = 10 + 0.5 * x
    y_normal = y_true_mean + rng.normal(loc=0, scale=0.5 + 0.5 * x, size=x.shape[0])

    # 拟合
    quantiles = [0.05, 0.5, 0.95]
    predictions = {}
    out_bounds_predictions = np.zeros_like(
        y_true_mean, dtype=np.bool_
    )  # zeros_like-与给定数组具有相同形状和数据类型的零数组
    for quantile in quantiles:
        qr = linear_model.QuantileRegressor(quantile=quantile, alpha=0)
        y_pred = qr.fit(X, y_normal).predict(X)
        predictions[quantile] = y_pred

        if quantile == min(quantiles):
            out_bounds_predictions = np.logical_or(
                out_bounds_predictions, y_pred >= y_normal
            )  #  logical_or-在两个布尔数组之间执行逻辑或运算
        elif quantile == max(quantiles):
            out_bounds_predictions = np.logical_or(
                out_bounds_predictions, y_pred <= y_normal
            )

    # 可视化
    for quantile, y_pred in predictions.items():
        plt.plot(X, y_pred, label=f"Quantile: {quantile}")

    plt.scatter(
        x[out_bounds_predictions],
        y_normal[out_bounds_predictions],
        color="black",
        marker="+",
        alpha=0.5,
        label="Outside interval",
    )
    plt.scatter(
        x[~out_bounds_predictions],
        y_normal[~out_bounds_predictions],
        color="black",
        alpha=0.5,
        label="Inside interval",
    )

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    _ = plt.title("Quantiles of heteroscedastic Normal distributed target")


if __name__ == "__main__":
    # Ordinary_LeastSquares_LinearRegression()  # OLS
    # NonNegative_LeastSquares_LinearRegression()  # NNLS
    # Ridge_Regression(alpha=0.5)  # Ridge
    # Ridge_Regression_CrossValidation()  # RidgeCV
    # Lasso_Regression(alpha=0.1)  # Lasso / LassoCV
    # MultiTask_Lasso_Regression(alpha=0.5)  # multi-task Lasso
    # ElasticNet(alpha=0.08, l1_ratio=0.5)  # ElasticNet / ElasticNet
    # LARSLasso_Regression(alpha=0.1)  # LARSLasso / LARSLasso
    # Orthogonal_Matching_Pursuit()   # OMP
    # BayesianRidge_Regression()  # BayesianRidge
    # Automatic_Relevance_Determination()  # ARD
    # Logistic_Regression(C=0.1, penalty="l2")  # Logistic / LogisticRegressionCV
    # GeneralizedLinearModels()  # GLM
    # RANdom_SAmple_Consensus()  # RANSAC
    # TheilSen_Regression()  # Theil_Sen
    Quantile_Regressor()  # Quantile
    plt.show()
