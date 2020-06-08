# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import time

import joblib
import numpy as np
import pandas as pd
from comet_ml import Experiment, Optimizer
from lightgbm import LGBMRegressor
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from ..utils.descriptornames import *
from ..utils.utils import augment_data, plot_predictions, read_pickle

# optimizer id 23b1617dc24a4fc48198650b1e4cc46f, took f46c00e7621147c7b129ff78e7ff4ad7 because the loss is here now greater than the stdev, which makes it seem more stable
PARAMETERS_01 = {
    "colsample_bytree": 0.20032975594543911,
    "max_depth": 30,
    "min_child_weight": 0.004286551076851089,
    "n_estimators": 1093,
    "num_leaves": 423,
    "reg_alpha": 1.520322189905922e-4,
    "reg_lambda": 0.056600636753530116,
    "subsample": 0.2339942750715971,
}

# optimizer id 11b4e66179b0472a8460e53918dbd5ec, parameters from bb425c091cf64643a932bac757d6a939
PARAMETERS_MEDIAN = {
    "colsample_bytree": 0.680864608679754,
    "max_depth": 49,
    "min_child_weight": 0.018165986208942688,
    "n_estimators": 2072,
    "num_leaves": 314,
    "reg_alpha": 0.013653685526724718,
    "reg_lambda": 4.720279281103699e-5,
    "subsample": 0.5006617953729169,
}


PARAMETERS_MEAN = {
    "colsample_bytree": 0.680864608679754,
    "max_depth": 49,
    "min_child_weight": 0.018165986208942688,
    "n_estimators": 2072,
    "num_leaves": 314,
    "reg_alpha": 0.013653685526724718,
    "reg_lambda": 4.720279281103699e-5,
    "subsample": 0.5006617953729169,
}

# optimizer id b1941f3a465b4f1bb14dce011cf04e66
PARAMETERS_09 = {
    "colsample_bytree": 0.46648360363558816,
    "max_depth": 34,
    "min_child_weight": 0.004943915466576356,
    "n_estimators": 4925,
    "num_leaves": 463,
    "reg_alpha": 2.274052930007024e-4,
    "reg_lambda": 0.006378438725066921,
    "subsample": 0.6830057873325602,
}

RANDOM_SEED = int(84996)

# CHEMICAL_FEATURES = (
#     metalcenter_descriptors
#     + functionalgroup_descriptors
#     + linker_descriptors
#     + mol_desc
#     + summed_functionalgroup_descriptors
#     + summed_linker_descriptors
#     + summed_metalcenter_descriptors
# )


CHEMICAL_FEATURES = [
    "mc_CRY-chi-0-all",
    "mc_CRY-chi-1-all",
    "mc_CRY-chi-2-all",
    "mc_CRY-chi-3-all",
    "mc_CRY-Z-0-all",
    "mc_CRY-Z-1-all",
    "mc_CRY-Z-2-all",
    "mc_CRY-Z-3-all",
    "mc_CRY-I-1-all",
    "mc_CRY-I-2-all",
    "mc_CRY-I-3-all",
    "mc_CRY-T-0-all",
    "mc_CRY-T-1-all",
    "mc_CRY-T-2-all",
    "mc_CRY-T-3-all",
    "mc_CRY-S-0-all",
    "mc_CRY-S-1-all",
    "mc_CRY-S-2-all",
    "mc_CRY-S-3-all",
    "D_mc_CRY-chi-1-all",
    "D_mc_CRY-chi-2-all",
    "D_mc_CRY-chi-3-all",
    "D_mc_CRY-Z-1-all",
    "D_mc_CRY-Z-2-all",
    "D_mc_CRY-Z-3-all",
    "D_mc_CRY-T-1-all",
    "D_mc_CRY-T-2-all",
    "D_mc_CRY-T-3-all",
    "D_mc_CRY-S-1-all",
    "D_mc_CRY-S-2-all",
    "D_mc_CRY-S-3-all",
    "func-chi-0-all",
    "func-chi-1-all",
    "func-chi-2-all",
    "func-chi-3-all",
    "func-Z-0-all",
    "func-Z-1-all",
    "func-Z-2-all",
    "func-Z-3-all",
    "func-I-1-all",
    "func-I-2-all",
    "func-I-3-all",
    "func-T-0-all",
    "func-T-1-all",
    "func-T-2-all",
    "func-T-3-all",
    "func-S-0-all",
    "func-S-1-all",
    "func-S-2-all",
    "func-S-3-all",
    "func-alpha-0-all",
    "func-alpha-1-all",
    "func-alpha-2-all",
    "func-alpha-3-all",
    "D_func-chi-1-all",
    "D_func-chi-2-all",
    "D_func-chi-3-all",
    "D_func-Z-1-all",
    "D_func-Z-2-all",
    "D_func-Z-3-all",
    "D_func-T-1-all",
    "D_func-T-2-all",
    "D_func-T-3-all",
    "D_func-S-2-all",
    "D_func-S-3-all",
    "D_func-alpha-1-all",
    "D_func-alpha-2-all",
    "D_func-alpha-3-all",
    "f-lig-chi-0",
    "f-lig-chi-1",
    "f-lig-chi-2",
    "f-lig-chi-3",
    "f-lig-Z-0",
    "f-lig-Z-1",
    "f-lig-Z-2",
    "f-lig-Z-3",
    "f-lig-I-0",
    "f-lig-I-1",
    "f-lig-I-2",
    "f-lig-I-3",
    "f-lig-T-0",
    "f-lig-T-1",
    "f-lig-T-2",
    "f-lig-T-3",
    "f-lig-S-0",
    "f-lig-S-1",
    "f-lig-S-2",
    "f-lig-S-3",
    "lc-chi-0-all",
    "lc-chi-1-all",
    "lc-chi-2-all",
    "lc-chi-3-all",
    "lc-Z-0-all",
    "lc-Z-1-all",
    "lc-Z-2-all",
    "lc-Z-3-all",
    "lc-I-2-all",
    "lc-I-3-all",
    "lc-T-0-all",
    "lc-T-1-all",
    "lc-T-2-all",
    "lc-T-3-all",
    "lc-S-3-all",
    "lc-alpha-0-all",
    "lc-alpha-1-all",
    "lc-alpha-2-all",
    "lc-alpha-3-all",
    "D_lc-chi-2-all",
    "D_lc-chi-3-all",
    "D_lc-Z-1-all",
    "D_lc-Z-2-all",
    "D_lc-Z-3-all",
    "D_lc-T-1-all",
    "D_lc-T-2-all",
    "D_lc-T-3-all",
    "D_lc-alpha-1-all",
    "D_lc-alpha-2-all",
    "D_lc-alpha-3-all",
    "tertiary_amide_sum",
    "ester_sum",
    "carbonyl_sum",
    "logP_sum",
    "MR_sum",
    "aromatic_rings_sum",
    "dbonds_sum",
    "abonds_sum",
    "tertiary_amide_mean",
    "ester_mean",
    "carbonyl_mean",
    "logP_mean",
    "MR_mean",
    "aromatic_rings_mean",
    "dbonds_mean",
    "abonds_mean",
    "sum-func-chi-0-all",
    "sum-func-chi-1-all",
    "sum-func-chi-2-all",
    "sum-func-chi-3-all",
    "sum-func-Z-0-all",
    "sum-func-Z-1-all",
    "sum-func-Z-2-all",
    "sum-func-Z-3-all",
    "sum-func-I-0-all",
    "sum-func-I-1-all",
    "sum-func-I-2-all",
    "sum-func-I-3-all",
    "sum-func-T-0-all",
    "sum-func-T-1-all",
    "sum-func-T-2-all",
    "sum-func-T-3-all",
    "sum-func-S-0-all",
    "sum-func-S-1-all",
    "sum-func-S-2-all",
    "sum-func-S-3-all",
    "sum-func-alpha-0-all",
    "sum-func-alpha-1-all",
    "sum-func-alpha-2-all",
    "sum-func-alpha-3-all",
    "sum-D_func-chi-1-all",
    "sum-D_func-chi-2-all",
    "sum-D_func-chi-3-all",
    "sum-D_func-Z-1-all",
    "sum-D_func-Z-2-all",
    "sum-D_func-Z-3-all",
    "sum-D_func-T-1-all",
    "sum-D_func-T-2-all",
    "sum-D_func-T-3-all",
    "sum-D_func-S-1-all",
    "sum-D_func-S-2-all",
    "sum-D_func-S-3-all",
    "sum-D_func-alpha-1-all",
    "sum-D_func-alpha-2-all",
    "sum-D_func-alpha-3-all",
    "sum-f-lig-chi-0",
    "sum-f-lig-chi-1",
    "sum-f-lig-chi-2",
    "sum-f-lig-chi-3",
    "sum-f-lig-Z-0",
    "sum-f-lig-Z-1",
    "sum-f-lig-Z-2",
    "sum-f-lig-Z-3",
    "sum-f-lig-I-0",
    "sum-f-lig-I-1",
    "sum-f-lig-I-2",
    "sum-f-lig-I-3",
    "sum-f-lig-T-0",
    "sum-f-lig-T-1",
    "sum-f-lig-T-2",
    "sum-f-lig-T-3",
    "sum-f-lig-S-0",
    "sum-f-lig-S-1",
    "sum-f-lig-S-2",
    "sum-f-lig-S-3",
    "sum-lc-chi-0-all",
    "sum-lc-chi-1-all",
    "sum-lc-chi-2-all",
    "sum-lc-chi-3-all",
    "sum-lc-Z-0-all",
    "sum-lc-Z-1-all",
    "sum-lc-Z-2-all",
    "sum-lc-Z-3-all",
    "sum-lc-I-0-all",
    "sum-lc-I-1-all",
    "sum-lc-I-2-all",
    "sum-lc-I-3-all",
    "sum-lc-T-0-all",
    "sum-lc-T-1-all",
    "sum-lc-T-2-all",
    "sum-lc-T-3-all",
    "sum-lc-S-0-all",
    "sum-lc-S-1-all",
    "sum-lc-S-2-all",
    "sum-lc-S-3-all",
    "sum-lc-alpha-0-all",
    "sum-lc-alpha-1-all",
    "sum-lc-alpha-2-all",
    "sum-lc-alpha-3-all",
    "sum-D_lc-chi-1-all",
    "sum-D_lc-chi-2-all",
    "sum-D_lc-chi-3-all",
    "sum-D_lc-Z-1-all",
    "sum-D_lc-Z-2-all",
    "sum-D_lc-Z-3-all",
    "sum-D_lc-T-1-all",
    "sum-D_lc-T-2-all",
    "sum-D_lc-T-3-all",
    "sum-D_lc-S-1-all",
    "sum-D_lc-S-2-all",
    "sum-D_lc-S-3-all",
    "sum-D_lc-alpha-1-all",
    "sum-D_lc-alpha-2-all",
    "sum-D_lc-alpha-3-all",
    "sum-mc_CRY-chi-0-all",
    "sum-mc_CRY-chi-1-all",
    "sum-mc_CRY-chi-2-all",
    "sum-mc_CRY-chi-3-all",
    "sum-mc_CRY-Z-0-all",
    "sum-mc_CRY-Z-1-all",
    "sum-mc_CRY-Z-2-all",
    "sum-mc_CRY-Z-3-all",
    "sum-mc_CRY-I-0-all",
    "sum-mc_CRY-I-1-all",
    "sum-mc_CRY-I-2-all",
    "sum-mc_CRY-I-3-all",
    "sum-mc_CRY-T-0-all",
    "sum-mc_CRY-T-1-all",
    "sum-mc_CRY-T-2-all",
    "sum-mc_CRY-T-3-all",
    "sum-mc_CRY-S-0-all",
    "sum-mc_CRY-S-1-all",
    "sum-mc_CRY-S-2-all",
    "sum-mc_CRY-S-3-all",
    "sum-D_mc_CRY-chi-1-all",
    "sum-D_mc_CRY-chi-2-all",
    "sum-D_mc_CRY-chi-3-all",
    "sum-D_mc_CRY-Z-1-all",
    "sum-D_mc_CRY-Z-2-all",
    "sum-D_mc_CRY-Z-3-all",
    "sum-D_mc_CRY-T-1-all",
    "sum-D_mc_CRY-T-2-all",
    "sum-D_mc_CRY-T-3-all",
    "sum-D_mc_CRY-S-1-all",
    "sum-D_mc_CRY-S-2-all",
    "sum-D_mc_CRY-S-3-all",
]

AUGMENT_DICT = (
    "/Users/kevinmaikjablonka/Dropbox (LSMO)/proj75_mofcolor/ml/data/augment_dict.pkl"
)


def process_data(augment=False):
    df_train = pd.read_csv(
        "/Users/kevinmaikjablonka/Dropbox (LSMO)/proj75_mofcolor/ml/data/development_set.csv"
    )
    df_test = pd.read_csv(
        "/Users/kevinmaikjablonka/Dropbox (LSMO)/proj75_mofcolor/ml/data/holdout_set.csv"
    )

    if augment:
        augment_dict = read_pickle(AUGMENT_DICT)
        df_train = augment_data(df_train, augment_dict)

    X_train = df_train[CHEMICAL_FEATURES]
    X_test = df_test[CHEMICAL_FEATURES]

    y_train = df_train[["r", "g", "b"]].values / 255
    y_test = df_test[["r", "g", "b"]].values / 255

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return scaler, None, X_train, X_test, y_train, y_test, df_test


def fit(x_train, y_train, parameters_01, parameters_median, parameters_09):
    regressor_mean = MultiOutputRegressor(LGBMRegressor(**parameters_median))
    regressor_mean.fit(x_train, y_train)

    regressor_median = MultiOutputRegressor(
        LGBMRegressor(objective="quantile", alpha=0.5, **parameters_median)
    )
    regressor_median.fit(x_train, y_train)

    regressor_0_1 = MultiOutputRegressor(
        LGBMRegressor(objective="quantile", alpha=0.1, **parameters_01)
    )
    regressor_0_1.fit(x_train, y_train)

    regressor_0_9 = MultiOutputRegressor(
        LGBMRegressor(objective="quantile", alpha=0.9, **parameters_09)
    )
    regressor_0_9.fit(x_train, y_train)

    return regressor_mean, regressor_median, regressor_0_1, regressor_0_9


def main():
    STARTTIME = time.strftime("run_%Y_%m_%d_%H_%M_%s")
    scaler, yscaler, X_train, X_test, y_train, y_test, df_test = process_data()

    joblib.dump(scaler, "scaler_" + STARTTIME + ".joblib")

    np.save("X_train_" + STARTTIME + ".npy", X_train)
    np.save("X_test_" + STARTTIME + ".npy", X_test)
    np.save("y_train_" + STARTTIME + ".npy", y_train)
    np.save("y_test_" + STARTTIME + ".npy", y_test)
    np.save("y_names_" + STARTTIME + ".npy", df_test["color_cleaned"])

    experiment = Experiment(
        api_key=os.environ["COMET_API_KEY"], project_name="color-ml"
    )

    experiment.log_asset("scaler_" + STARTTIME + ".joblib")

    experiment.log_asset("X_train_" + STARTTIME + ".npy")
    experiment.log_asset("X_test_" + STARTTIME + ".npy")
    experiment.log_asset("y_train_" + STARTTIME + ".npy")
    experiment.log_asset("y_test_" + STARTTIME + ".npy")
    experiment.log_asset("y_names_" + STARTTIME + ".npy")

    experiment.log_parameters(PARAMETERS_01)
    experiment.log_parameters(PARAMETERS_MEDIAN)
    experiment.log_parameters(PARAMETERS_09)

    with experiment.train():
        regressor_mean, regressor_median, regressor_0_1, regressor_0_9 = fit(
            X_train, y_train, PARAMETERS_01, PARAMETERS_MEDIAN, PARAMETERS_09
        )

    joblib.dump(regressor_mean, "regressor_mean" + STARTTIME + ".joblib")
    joblib.dump(regressor_median, "regressor_median" + STARTTIME + ".joblib")
    joblib.dump(regressor_0_1, "regressor_0_1" + STARTTIME + ".joblib")
    joblib.dump(regressor_0_9, "regressor_0_9" + STARTTIME + ".joblib")

    experiment.log_asset("regressor_mean" + STARTTIME + ".joblib")
    experiment.log_asset("regressor_median" + STARTTIME + ".joblib")
    experiment.log_asset("regressor_0_1" + STARTTIME + ".joblib")
    experiment.log_asset("regressor_0_9" + STARTTIME + ".joblib")

    with experiment.test():
        mean_predict = regressor_mean.predict(X_test)
        r2 = r2_score(y_test, mean_predict)
        mae = mean_absolute_error(y_test, mean_predict)
        mse = mean_squared_error(y_test, mean_predict)

        experiment.log_metric("r2_score", r2)
        experiment.log_metric("mae", mae)
        experiment.log_metric("mse", mse)

        plot_predictions(
            mean_predict * 255,
            y_test * 255,
            df_test["color_cleaned"].values,
            outname="mean_" + STARTTIME + ".png",
        )
        experiment.log_image("mean_" + STARTTIME + ".png")

        median_predict = regressor_median.predict(X_test)

        plot_predictions(
            median_predict * 255,
            y_test * 255,
            df_test["color_cleaned"].values,
            outname="median_" + STARTTIME + ".png",
        )

        experiment.log_image("median_" + STARTTIME + ".png")

        r_0_1_predict = regressor_0_1.predict(X_test)

        plot_predictions(
            r_0_1_predict * 255,
            y_test * 255,
            df_test["color_cleaned"].values,
            outname="quantile_0_1_" + STARTTIME + ".png",
        )
        experiment.log_image("quantile_0_1_" + STARTTIME + ".png")

        r_0_9_predict = regressor_0_9.predict(X_test)

        plot_predictions(
            r_0_9_predict * 255,
            y_test * 255,
            df_test["color_cleaned"].values,
            outname="quantile_0_9_" + STARTTIME + ".png",
        )
        experiment.log_image("quantile_0_9_" + STARTTIME + ".png")


if __name__ == "__main__":
    main()
