from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1314)

"""
1.1.1.2. Ordinary Least Squares Complexity
1.1.2.3. Ridge Complexity
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

    Notes
    -----
    使用L2正则化引入惩罚项, 减少模型复杂度避免过拟合
    alpha越大, shrinkage越强, 系数共线性越强
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

    Notes
    -----
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


if __name__ == "__main__":
    Ordinary_Least_Squares_Linear_Regression()  # OLS
    NonNegative_Least_Squares_Linear_Regression()  # NNLS
    Ridge_Regression(alpha=0.5)  # Ridge
    Ridge_Regression_CrossValidation()  # RidgeCV
    Lasso(alpha=0.5)  # Lasso
