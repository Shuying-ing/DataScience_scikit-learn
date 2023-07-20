"""1.1.12. Generalized Linear Models

    Poisson regression and non-normal loss
    - Goal: redict the expected frequency of claims following car accidents

Lessons learned
-----
- The least squares loss of the Ridge regression model seems to cause this model to be badly calibrated.
    In particular, it tends to underestimate the risk and can even predict invalid negative frequencies.
    # if (~mask).any():
- Using the Poisson loss with a log-link can correct these problems and lead to a well-calibrated linear model.
- Traditional regression metrics such as MSE and MAE are hard to meaningfully interpret on count values with many zeros.
"""

from sklearn.datasets import fetch_openml  # fetch_openml-调用OpenML接口
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline  # make_pipeline-自动为每一步骤指定名称
from sklearn.pipeline import Pipeline  # Pipeline-自动为每一步骤指定名称
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_dataset():
    """Load French Motor Third-Party Liability Claims dataset

    Notes
    -----
    Claim : the request made by a policyholder to the insurer to compensate for a loss covered by the insurance.
    Exposure : the duration of the insurance coverage of a given policy, in years.
    """

    df = fetch_openml(data_id=41214, as_frame=True).frame  # 678013 × 12
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    # 可视化——直方图
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(16, 4))
    ax0.set_title("Number of claims")
    _ = df["ClaimNb"].hist(bins=30, log=True, ax=ax0)
    ax1.set_title("Exposure in years")
    _ = df["Exposure"].hist(bins=30, log=True, ax=ax1)
    ax2.set_title("Frequency (number of claims per year)")
    _ = df["Frequency"].hist(bins=30, log=True, ax=ax2)

    df_train, df_test = train_test_split(df, test_size=0.33, random_state=0)

    return df_train, df_test


def model(df_train, df_test):
    """Model with Baseline-DummyRegressor & GLM-Ridge/Poission"""

    # 预处理
    log_scale_transformer = make_pipeline(
        FunctionTransformer(np.log, validate=False), StandardScaler()
    )
    linear_model_preprocessor = ColumnTransformer(
        [
            ("passthrough_numeric", "passthrough", ["BonusMalus"]),
            (
                "binned_numeric",
                KBinsDiscretizer(n_bins=10, subsample=int(2e5), random_state=0),
                ["VehAge", "DrivAge"],
            ),  # KBinsDiscretizer-将连续数据分成等宽的bins
            ("log_scaled_numeric", log_scale_transformer, ["Density"]),
            (
                "onehot_categorical",
                OneHotEncoder(),
                ["VehBrand", "VehPower", "VehGas", "Region", "Area"],
            ),
        ],
        remainder="drop",
    )  # 对某些列预处理

    # 拟合——baselineModel
    dummy = Pipeline(
        [
            ("preprocessor", linear_model_preprocessor),
            ("regressor", DummyRegressor(strategy="mean")),
        ]
    ).fit(
        df_train, df_test["Frequency"], regressor__sample_weight=df_train["Exposure"]
    )  # 根据自定义字段对损失函数加权

    # 拟合——Ridge_glm
    ridge_glm = Pipeline(
        [
            ("preprocessor", linear_model_preprocessor),
            ("regressor", Ridge(alpha=1e-6)),
        ]
    ).fit(
        df_train, df_train["Frequency"], regressor__sample_weight=df_train["Exposure"]
    )

    # 拟合——poisson_glm
    poisson_glm = Pipeline(
        [
            ("preprocessor", linear_model_preprocessor),
            ("regressor", PoissonRegressor(alpha=1e-12, solver="newton-cholesky")),
        ]
    ).fit(
        df_train, df_train["Frequency"], regressor__sample_weight=df_train["Exposure"]
    )

    return dummy, ridge_glm, poisson_glm


def score_estimator(estimator, df_test):
    """Score an estimator on the test set"""

    y_pred = estimator.predict(df_test)
    # 测试
    print(
        "MSE : %.3f"
        % mean_squared_error(
            df_test["Frequency"], y_pred, sample_weight=df_test["Exposure"]
        )
    )
    print(
        "MAE: %.3f"
        % mean_absolute_error(
            df_test["Frequency"], y_pred, sample_weight=df_test["Exposure"]
        )
    )

    mask = y_pred > 0
    if (~mask).any():  # any()-判断是否有任何一个元素=True（y_pred中有任何一个元素≤0）
        n_masked, n_samples = (~mask).sum(), mask.shape[0]
        print(
            "WARNING: Estimator yields invalid, non-positive predictions "
            f" for {n_masked} samples out of {n_samples}. These predictions "
            "are ignored when computing the Poisson deviance."
        )
    print(
        "mean Poisson deviance: %.3f"
        % mean_poisson_deviance(
            df_test["Frequency"][mask],
            y_pred[mask],
            sample_weight=df_test["Exposure"][mask],
        )
    )


if __name__ == "__main__":

    # 准备数据
    df_train, df_test = load_dataset()
    # 拟合
    dummy, ridge_glm, poisson_glm = model(df_train, df_test)
    # 测试
    print("Dummy evaluation:")
    score_estimator(dummy, df_test)
    print("Ridge evaluation:")
    score_estimator(ridge_glm, df_test)
    print("PoissonRegressor evaluation:")
    score_estimator(poisson_glm, df_test)

    plt.show()
