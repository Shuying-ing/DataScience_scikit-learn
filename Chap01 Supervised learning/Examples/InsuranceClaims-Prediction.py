"""1.1.12. Generalized Linear Models

    Tweedie regression on insurance claims
    - Goal: predict the expected value (the mean) of the total claim amount per exposure unit
    - illustrates the use of Poisson, Gamma and Tweedie regression

    Two methods to predict total claim amount:
    1. Model the number of claims with a Poisson distribution, and the average claim amount per claim as a Gamma distribution
        and multiply the predictions of both in order to get the total claim amount.
    2. Model the total claim amount per exposure directly, typically with a Tweedie distribution

Lessons learned
-----
- tweedie distributions: https://zhuanlan.zhihu.com/p/438439576

-----
********** A Very Very Very Detailed Lesson !!!!!!!!!!!!!

"""

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.linear_model import (
    PoissonRegressor,
    GammaRegressor,
    TweedieRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_tweedie_deviance,
    auc,
)
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_dataset(n_samples=None):
    """Load the French Motor Third-Party Liability Claims dataset.

    Parameters
    ----------
    n_samples: int, default=None
      number of samples to select (for faster run time). Full dataset has
      678013 samples.
    """

    df_freq = fetch_openml(data_id=41214, as_frame=True).data  # 678013 x 12
    df_freq["IDpol"] = df_freq["IDpol"].astype(int)
    df_freq.set_index("IDpol", inplace=True)  # 678013 x 11 去掉了序号列

    df_sev = fetch_openml(data_id=41215, as_frame=True).data
    df_sev = df_sev.groupby("IDpol").sum()  # groupy.sum()-按IDpol列分组，其他列求和  | 678013 x 1

    df = df_freq.join(df_sev, how="left")  # 678013 x 12
    df["ClaimAmount"].fillna(0, inplace=True)  # fillna-填充缺失值为0

    # unquote string fields
    for column_name in df.columns[df.dtypes.values == object]:  # 查找数据类型为字符串的列
        df[column_name] = df[column_name].str.strip("'")  # 去掉每个字符串的开头和结尾的单引号

    return df.iloc[:n_samples]  # iloc-切片（返回前n_samples行）


def preprocess_data(df):
    """Preprocess data"""

    # Note: filter out claims with zero amount, as the severity model
    # requires strictly positive target values.
    df.loc[(df["ClaimAmount"] == 0) & (df["ClaimNb"] >= 1), "ClaimNb"] = 0

    # Correct for unreasonable observations (that might be data error)
    # and a few exceptionally large claim amounts
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(upper=1)
    df["ClaimAmount"] = df["ClaimAmount"].clip(upper=200000)

    # 预处理
    log_scale_transformer = make_pipeline(
        FunctionTransformer(func=np.log), StandardScaler()
    )
    column_trans = ColumnTransformer(
        [
            (
                "binned_numeric",
                KBinsDiscretizer(n_bins=10),  # subsample=int(2e5), random_state=0
                ["VehAge", "DrivAge"],
            ),
            (
                "onehot_categorical",
                OneHotEncoder(),
                ["VehBrand", "VehPower", "VehGas", "Region", "Area"],
            ),
            ("passthrough_numeric", "passthrough", ["BonusMalus"]),
            ("log_scaled_numeric", log_scale_transformer, ["Density"]),
        ],
        remainder="drop",
    )
    X = column_trans.fit_transform(df)

    # Method1 关心的数据
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    df["AvgClaimAmount"] = df["ClaimAmount"] / np.fmax(df["ClaimNb"], 1)

    # Method2 关心的数据
    df["PurePremium"] = df["ClaimAmount"] / df["Exposure"]
    # with pd.option_context("display.max_columns", 15):
    #     print(df[df.ClaimAmount > 0].head())

    df_train, df_test, X_train, X_test = train_test_split(df, X, random_state=0)
    # print("df_train", type(df_train))  # <class 'pandas.core.frame.DataFrame'>
    # print("X_train", type(X_train))  # <class 'scipy.sparse.csr.csr_matrix'>
    # print(df_train.shape)  # (508509, 12)
    # print(X_train.shape)  # (508509, 75)

    return df_train, df_test, X_train, X_test


def model(df_train, X_train):
    """Model

    Notes
    -----
    Method 1:
        - Frequency model with Poisson distribution
        - Mean claim amount/Severity Model with Gamma distribution
    Method 2:
        - Pure Premium Model with Tweedie distribution
    """

    # 拟合——Method 1 Frequency model
    glm_freq = PoissonRegressor(alpha=1e-4)  # solver="newton-cholesky"
    glm_freq.fit(X_train, df_train["Frequency"], sample_weight=df_train["Exposure"])

    # 拟合——Method 1 Mean claim amount/Severity Model
    mask_train = df_train["ClaimAmount"] > 0
    glm_sev = GammaRegressor(alpha=10.0)  # solver="newton-cholesky"
    glm_sev.fit(
        X_train[mask_train.values],
        df_train.loc[mask_train, "AvgClaimAmount"],
        sample_weight=df_train.loc[mask_train, "ClaimNb"],
    )

    # 拟合——Method 2 Pure Premium Model
    glm_pure_premium = TweedieRegressor(
        power=1.9, alpha=0.1
    )  # solver="newton-cholesky"
    glm_pure_premium.fit(
        X_train, df_train["PurePremium"], sample_weight=df_train["Exposure"]
    )

    return glm_freq, glm_sev, glm_pure_premium


def score_estimator(
    estimator,
    X_train,
    X_test,
    df_train,
    df_test,
    target,
    weights,
    tweedie_powers=None,
):
    """Evaluate an estimator on train and test sets with different metrics"""

    metrics = [
        ("D² explained", None),  # Use default scorer if it exists
        ("MAE", mean_absolute_error),
        ("MSE", mean_squared_error),
    ]
    if tweedie_powers:
        metrics += [
            (
                "mean Tweedie dev p={:.4f}".format(power),
                partial(mean_tweedie_deviance, power=power),
            )  # partial-使用函数创建一个新函数
            for power in tweedie_powers
        ]

    res = []
    # 针对训练集和测试集
    for subset_label, X, df in [
        ("train", X_train, df_train),
        ("test", X_test, df_test),
    ]:
        y, _weights = df[target], df[weights]

        # 针对不同测度
        for score_label, metric in metrics:
            # 两种方法下的预测结果
            if (
                isinstance(estimator, tuple) and len(estimator) == 2
            ):  # Method1, isinstance-检查是否为元组
                est_freq, est_sev = estimator
                y_pred = est_freq.predict(X) * est_sev.predict(X)
            else:  # Method2
                y_pred = estimator.predict(X)

            if metric is None:
                if not hasattr(estimator, "score"):  # hasattr-检查tuple中是否有“score”属性
                    continue
                score = estimator.score(X, y, sample_weight=_weights)
            else:
                score = metric(y, y_pred, sample_weight=_weights)

            res.append({"subset": subset_label, "metric": score_label, "score": score})

    res = (
        pd.DataFrame(res)
        .set_index(["metric", "subset"])
        .score.unstack(-1)
        .round(4)
        .loc[:, ["train", "test"]]
    )  # set_index-设置索引列, score.unstack(-1)-设置score为新的列

    return res


def lorenz_curve(y_true, y_pred, exposure):
    """Visualize results with lorenz curve"""

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))

    return cumulated_samples, cumulated_claim_amount


if __name__ == "__main__":
    # 准备数据
    df = load_dataset()
    df_train, df_test, X_train, X_test = preprocess_data(df)

    # 拟合
    glm_freq, glm_sev, glm_pure_premium = model(df_train, X_train)

    # """
    #     测试 Method 1 —— 两个模型
    # """
    # # 单独测试 —— Method 1 Frequency
    # scores = score_estimator(
    #     glm_freq,
    #     X_train,
    #     X_test,
    #     df_train,
    #     df_test,
    #     target="Frequency",
    #     weights="Exposure",
    # )
    # # print("\nEvaluation of PoissonRegressor on target Frequency")
    # # print(scores)
    # # 单独测试 —— Method 1 AvgClaimAmount
    # mask_train = df_train["ClaimAmount"] > 0
    # mask_test = df_test["ClaimAmount"] > 0
    # scores = score_estimator(
    #     glm_sev,
    #     X_train[mask_train.values],
    #     X_test[mask_test.values],
    #     df_train[mask_train],
    #     df_test[mask_test],
    #     target="AvgClaimAmount",
    #     weights="ClaimNb",
    # )
    # # print("\nEvaluation of GammaRegressor on target AvgClaimAmount")
    # # print(scores)

    """
        测试 Method 1/2
    """
    tweedie_powers = [1.5, 1.7, 1.8, 1.9, 1.99, 1.999, 1.9999]
    # 测试 —— Method 1
    scores_product_model = score_estimator(
        (glm_freq, glm_sev),
        X_train,
        X_test,
        df_train,
        df_test,
        target="PurePremium",
        weights="Exposure",
        tweedie_powers=tweedie_powers,
    )
    # 测试 —— Method 2
    scores_glm_pure_premium = score_estimator(
        glm_pure_premium,
        X_train,
        X_test,
        df_train,
        df_test,
        target="PurePremium",
        weights="Exposure",
        tweedie_powers=tweedie_powers,
    )

    scores = pd.concat(
        [scores_product_model, scores_glm_pure_premium],
        axis=1,
        sort=True,
        keys=("Product Model", "TweedieRegressor"),
    )  # concat-拼接df
    print(
        "Evaluation of the Product Model and the Tweedie Regressor on target PurePremium"
    )
    with pd.option_context(
        "display.expand_frame_repr", False
    ):  # option_context-输出时所有列不截断地显示在一行
        print(scores)

    """
        可视化 Method 1/2
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    y_pred_product = glm_freq.predict(X_test) * glm_sev.predict(X_test)
    y_pred_total = glm_pure_premium.predict(X_test)
    for label, y_pred in [
        ("Frequency * Severity model", y_pred_product),
        ("Compound Poisson Gamma", y_pred_total),
    ]:
        ordered_samples, cum_claims = lorenz_curve(
            df_test["PurePremium"], y_pred, df_test["Exposure"]
        )
        gini = 1 - 2 * auc(ordered_samples, cum_claims)
        label += " (Gini index: {:.3f})".format(gini)
        ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)
    ax.set(
        title="Lorenz Curves",
        xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
        ylabel="Fraction of total claim amount",
    )
    ax.legend(loc="upper left")
    plt.show()
