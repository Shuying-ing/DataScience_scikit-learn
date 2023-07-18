"""1.1.2.2. Classification

    Common pitfalls in the interpretation of coefficients of linear models

Lessons learned
-----
- 特征系数必须在相同度量下才有讨论其重要性的意义, i.e., 需要standard-deviation
    model(flag_StandardedVariables)
    interpret_modelCoefficients()

- 特征相关性和最后特征系数的表现可能不同
    pairplot_data(), interpret_modelCoefficients()

- 系数稳定性可能受到特征关联性的影响
    check_coefficientsVariability(),
    check_coefficientsVariability_with_featureRemoved()

- 不同的模型下的结果可能大相径庭 1.系数稀疏性【Ridge/LASSO】 2.自适应regulation值【RidgeCV/LassoCV】
    model(model_type),
    plot_predictionResults(model_type)

- 通过检查交叉验证循环的折叠系数, 可以了解系数稳定性
    check_coefficientsVariability()

"""

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_validate
from sklearn.compose import make_column_transformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge, RidgeCV, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import median_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns


def load_dataset():
    """Load wage dataset"""

    survey = fetch_openml(data_id=534, as_frame=True)  # as_frame-保持数据为pd格式
    survey.data.info()  # info-数据类型/数量
    X = survey.data[survey.feature_names]
    X.describe(include="all")  # describe-描述数据统计特性
    X.head()
    # print(type(survey.target))  # <class 'pandas.core.series.Series'>
    # print(type(survey.target.values))  # <class 'numpy.ndarray'>
    y = survey.target.values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42
    )  # 固定random_state

    return X_train, X_test, y_train, y_test, X, y


def pairplot_data(X_train, y_train):
    """Visualize the relationships among data

    Notes
    -----
    发现wages长尾分布, 所以后边进行np.log10使其成为正态分布
    """

    train_dataset = X_train.copy()
    train_dataset.insert(0, "WAGE", y_train)  # wage-最关键的变量
    _ = sns.pairplot(train_dataset, kind="reg", diag_kind="kde")  # diag_kind-对角线上的图类型


def model(
    X_train,
    X_test,
    y_train,
    y_test,
    model_type,
    flag_StandardedVariables,
):
    """model

    Notes
    -----
    先数据预处理 = 对类别型数据独热编码 + 对数值型数据归一化(可选)
    再基于Ridge模型回归, (可选)Ridge模型参数alpha是否自适应
    """

    categorical_columns = [
        "RACE",
        "OCCUPATION",
        "SECTOR",
        "MARR",
        "UNION",
        "SEX",
        "SOUTH",
    ]
    numerical_columns = ["EDUCATION", "EXPERIENCE", "AGE"]

    if flag_StandardedVariables:
        preprocessor = make_column_transformer(
            (OneHotEncoder(drop="if_binary"), categorical_columns),
            (StandardScaler(), numerical_columns),  # 对数值型数据-标准化
        )
    else:
        preprocessor = make_column_transformer(
            (OneHotEncoder(drop="if_binary"), categorical_columns),
            remainder="passthrough",  # remainder-对其余变量的处理方法-保持不变
            verbose_feature_names_out=False,
        )  # OneHotEncoder-对非二元的类别型变量进行独热编码

    if model_type == "Ridge":
        model_whole = make_pipeline(
            preprocessor,
            TransformedTargetRegressor(
                regressor=Ridge(alpha=1e-10),
                func=np.log10,  # func-通过pairplot_data发现数据长尾
                inverse_func=sp.special.exp10,  # inverse_func-和func相对, 预测时转化为原特征空间
            ),
        )  # make_pipeline操作
    elif model_type == "RidgeCV":
        alphas = np.logspace(-10, 10, 21)  # 自适应alpha值
        model_whole = make_pipeline(
            preprocessor,
            TransformedTargetRegressor(
                regressor=RidgeCV(alphas=alphas),
                func=np.log10,
                inverse_func=sp.special.exp10,
            ),
        )
    elif model_type == "LassoCV":
        alphas = np.logspace(-10, 10, 21)
        model_whole = make_pipeline(
            preprocessor,
            TransformedTargetRegressor(
                regressor=LassoCV(alphas=alphas, max_iter=100_000),
                func=np.log10,
                inverse_func=sp.special.exp10,
            ),
        )

    # 拟合
    model_whole.fit(X_train, y_train)
    if model_type == "RidgeCV":
        print("alpha of RidgeCV: ", model_whole[-1].regressor_.alpha_)
    elif model_type == "LassoCV":
        print("alpha of LassoCV: ", model_whole[-1].regressor_.alpha_)
    # 测试
    mae_train = median_absolute_error(y_train, model_whole.predict(X_train))
    y_pred = model_whole.predict(X_test)
    mae_test = median_absolute_error(y_test, y_pred)
    scores = {
        "MedAE on training set": f"{mae_train:.2f} $/hour",
        "MedAE on testing set": f"{mae_test:.2f} $/hour",
    }  # 使用键值对存储结果，作为后边图表上的legend
    print(scores)

    return model_whole, y_pred, scores


def plot_predictionResults(model_type, y_test, y_pred, scores):
    """Visualize the prediction results"""

    _, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_test, y_pred)
    for name, score in scores.items():
        ax.plot([], [], label=f"{name}: {score}")  # 这种方式增加图例！！
    ax.legend(loc="upper left")
    if model_type == "Ridge":
        ax.set_title("Ridge model, small regularization")
    elif model_type == "RidgeCV":
        ax.set_title("Ridge model, optimum regularization")
    elif model_type == "LassoCV":
        ax.set_title("Lasso model, optimum regularization")
    plt.tight_layout()  # tight_layout-优化布局/padding


def interpret_modelCoefficients(model_type, model):
    """Interpret model coefficients

    Notes
    -----
    将系数乘以相关特征的标准偏差=将所有系数减少到相同的度量单位=将数值变量归一化到它们的标准差

    结果说明：
    - 在上边【pairplot】图中 —— wage随age增加而升高,
    - 在【可视化——归一化后的预测系数】图中 —— wage随age增加而降低, 因为这里是conditional dependencies【其他元素值都固定】
    """

    # print(model[:-1])  # Pipeline(ColumnTransformer, TransformedTargetRegressor)
    # print(model[-1])  # model(Pipeline)的最后一个对象, TransformedTargetRegressor
    feature_names = model[
        :-1
    ].get_feature_names_out()  # model最后一个对象是估计器xxxRegressor, 不包含特征名称, 所以需要排除

    # 可视化——预测系数
    coefs = pd.DataFrame(
        model[-1].regressor_.coef_,
        columns=["Coefficients"],
        index=feature_names,
    )  # DataFrame-一列name一列coef
    coefs.plot.barh(figsize=(9, 7))
    plt.axvline(x=0, color=".5")  # axvline-在x=0处绘制垂直线
    plt.xlabel("Raw coefficient values")
    plt.title("Ridge model, small regularization")
    plt.subplots_adjust(left=0.3)  # subplots_adjust-将左边的子图边缘向右移动30％的轴宽度

    # 可视化——标准差
    X_train_preprocessed = pd.DataFrame(
        model[:-1].transform(X_train), columns=feature_names
    )  # model[:-1]-数据经过了热编码对象, trasform-将数据集转化为模型的特征空间
    std = X_train_preprocessed.std(axis=0)  # axis=0-沿列计算标准差, i.e., 每个特征的
    std.plot.barh(figsize=(9, 7))  # .plot.barh-数据直接画条形图
    plt.xlabel("Std. dev. of feature values")
    plt.title("Feature ranges")
    plt.subplots_adjust(left=0.3)

    # 可视化——归一化后的预测系数
    coefs = pd.DataFrame(
        model[-1].regressor_.coef_ * X_train_preprocessed.std(axis=0),
        columns=["Coefficients importance"],
        index=feature_names,
    )  # 本质-系数×标准差
    coefs.plot.barh(figsize=(9, 7))
    plt.axvline(x=0, color=".5")
    plt.xlabel("Coefficient values corrected by the feature's std. dev.")
    if model_type == "Ridge":
        plt.suptitle("Ridge model, small regularization")
    elif model_type == "RidgeCV":
        plt.suptitle("Ridge model, optimum regularization")
    elif model_type == "LassoCV":
        plt.suptitle("Lasso model, optimum regularization")
    plt.subplots_adjust(left=0.3)


def check_coefficientsVariability(X, y, model_type, model):
    """Check the coefficient variability through cross-validation"""

    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
    cv_model = cross_validate(
        model,
        X,
        y,
        cv=cv,
        return_estimator=True,
        # n_jobs=2,
    )  # n_jobs=2-2个CPU核心并行运行作业, 但自己的电脑可能不行?

    feature_names = model[:-1].get_feature_names_out()
    coefs = pd.DataFrame(
        [
            est[-1].regressor_.coef_ * est[:-1].transform(X.iloc[train_idx]).std(axis=0)
            for est, (train_idx, _) in zip(cv_model["estimator"], cv.split(X, y))
        ],
        columns=feature_names,
    )  # 本质-系数×标准差, 只是遍历了每一个拆分情况; iloc-按位置选择行和列, 这里是行

    # 可视化
    plt.figure(figsize=(9, 7))
    sns.stripplot(
        data=coefs, orient="h", palette="gray", alpha=0.5
    )  # stripplot-带有散点图的分类变量
    sns.boxplot(
        data=coefs, orient="h", color="cyan", saturation=0.5, whis=10
    )  # boxplot-带有箱线图的分类变量
    plt.axvline(x=0, color=".5")
    plt.xlabel("Coefficient importance")
    plt.title("Coefficient importance and its variability")
    if model_type == "Ridge":
        plt.suptitle("Ridge model, small regularization")
    elif model_type == "RidgeCV":
        plt.suptitle("Ridge model, optimum regularization")
    elif model_type == "LassoCV":
        plt.suptitle("Lasso model, optimum regularization")
    plt.subplots_adjust(left=0.3)


def check_coefficientsVariability_with_featureRemoved(
    X, y, model_type, model, column_to_drop
):
    """Check the coefficient variability through cross-validation

    Notes
    -----
    当去除一些个变量时, 结果显然会发生变化
    """

    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
    cv_model = cross_validate(
        model,
        X.drop(columns=column_to_drop),
        y,
        cv=cv,
        return_estimator=True,
        # n_jobs=2,
    )

    feature_names = model[:-1].get_feature_names_out()
    coefs = pd.DataFrame(
        [
            est[-1].regressor_.coef_
            * est[:-1]
            .transform(X.drop(columns=column_to_drop).iloc[train_idx])
            .std(axis=0)
            for est, (train_idx, _) in zip(cv_model["estimator"], cv.split(X, y))
        ],
        columns=feature_names[:-1],
    )

    # 可视化
    plt.figure(figsize=(9, 7))
    sns.stripplot(data=coefs, orient="h", palette="gray", alpha=0.5)
    sns.boxplot(data=coefs, orient="h", color="cyan", saturation=0.5)
    plt.axvline(x=0, color=".5")
    plt.title("Coefficient importance and its variability")
    plt.xlabel("Coefficient importance")
    if model_type == "Ridge":
        plt.suptitle("Ridge model, small regularization, AGE dropped")
    elif model_type == "RidgeCV":
        plt.suptitle("Ridge model, optimum regularization, AGE dropped")
    elif model_type == "LassoCV":
        plt.suptitle("Lasso model, optimum regularization, AGE dropped")
    plt.subplots_adjust(left=0.3)


if __name__ == "__main__":
    flag_StandardedVariables = False  # 是否标准化数值型数据
    model_type = "LassoCV"  # Ridge/RidgeCV/LassoCV

    X_train, X_test, y_train, y_test, X, y = load_dataset()
    pairplot_data(X_train, y_train)
    model_whole, y_pred, scores = model(
        X_train, X_test, y_train, y_test, model_type, flag_StandardedVariables
    )
    plot_predictionResults(model_type, y_test, y_pred, scores)
    interpret_modelCoefficients(model_type, model_whole)
    check_coefficientsVariability(X, y, model_type, model_whole)
    if not flag_StandardedVariables:
        check_coefficientsVariability_with_featureRemoved(
            X, y, model_type, model_whole, ["AGE"]
        )  # flag_StandardedVariables会影响模型对象的系数是否存在啥的(regressor_)
    plt.show()
