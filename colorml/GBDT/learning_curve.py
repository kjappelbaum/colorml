# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import time

import joblib
import numpy as np
import pandas as pd
from colour.models import RGB_to_HSV
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

PARAMETERS = {
    "colsample_bytree": 0.38177744189426355,
    "max_depth": 28,
    "min_child_weight": 0.009519910628552617,
    "n_estimators": 1230,
    "num_leaves": 252,
    "reg_alpha": 0.0015621818443089072,
    "reg_lambda": 0.037839844391839766,
    "subsample": 0.3804212840372488,
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

    yscaler = StandardScaler()
    y_train = yscaler.fit_transform(RGB_to_HSV(df_train[["r", "g", "b"]]))
    y_test = yscaler.transform(RGB_to_HSV(df_test[["r", "g", "b"]]))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return scaler, yscaler, X_train, X_test, y_train, y_test, df_test


def fit(x_train, y_train, parameters):
    regressor_mean = MultiOutputRegressor(LGBMRegressor(**parameters))
    regressor_mean.fit(x_train, y_train)

    regressor_median = MultiOutputRegressor(
        LGBMRegressor(objective="quantile", alpha=0.5, **parameters)
    )
    regressor_median.fit(x_train, y_train)

    regressor_0_1 = MultiOutputRegressor(
        LGBMRegressor(objective="quantile", alpha=0.1, **parameters)
    )
    regressor_0_1.fit(x_train, y_train)

    regressor_0_9 = MultiOutputRegressor(
        LGBMRegressor(objective="quantile", alpha=0.9, **parameters)
    )
    regressor_0_9.fit(x_train, y_train)

    return regressor_mean, regressor_median, regressor_0_1, regressor_0_9


def main():
    STARTTIME = time.strftime("run_%Y_%m_%d_%H_%M_%s")
    scaler, yscaler, X_train, X_test, y_train, y_test, df_test = process_data()

    joblib.dump(scaler, "scaler_" + STARTTIME + ".joblib")

    joblib.dump(yscaler, "yscaler_" + STARTTIME + ".joblib")
    np.save("X_train_" + STARTTIME + ".npy", X_train)
    np.save("X_test_" + STARTTIME + ".npy", X_test)
    np.save("y_train_" + STARTTIME + ".npy", y_train)
    np.save("y_test_" + STARTTIME + ".npy", y_test)
    np.save("y_names_" + STARTTIME + ".npy", df_test["color_cleaned"])

    experiment = Experiment(
        api_key=os.environ["COMET_API_KEY"], project_name="color-ml"
    )

    experiment.log_asset("scaler_" + STARTTIME + ".joblib")

    experiment.log_asset("yscaler_" + STARTTIME + ".joblib")
    experiment.log_asset("X_train_" + STARTTIME + ".npy")
    experiment.log_asset("X_test_" + STARTTIME + ".npy")
    experiment.log_asset("y_train_" + STARTTIME + ".npy")
    experiment.log_asset("y_test_" + STARTTIME + ".npy")
    experiment.log_asset("y_names_" + STARTTIME + ".npy")

    experiment.log_parameters(PARAMETERS)

    with experiment.train():
        regressor_mean, regressor_median, regressor_0_1, regressor_0_9 = fit(
            X_train, y_train, PARAMETERS
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
            yscaler.inverse_transform(mean_predict),
            yscaler.inverse_transform(y_test),
            df_test["color_cleaned"].values,
            outname="mean_" + STARTTIME + ".png",
        )
        experiment.log_image("mean_" + STARTTIME + ".png")

        median_predict = regressor_median.predict(X_test)

        plot_predictions(
            yscaler.inverse_transform(median_predict),
            yscaler.inverse_transform(y_test),
            df_test["color_cleaned"].values,
            outname="median_" + STARTTIME + ".png",
        )

        experiment.log_image("median_" + STARTTIME + ".png")

        r_0_1_predict = regressor_0_1.predict(X_test)

        plot_predictions(
            yscaler.inverse_transform(r_0_1_predict),
            yscaler.inverse_transform(y_test),
            df_test["color_cleaned"].values,
            outname="quantile_0_1_" + STARTTIME + ".png",
        )
        experiment.log_image("quantile_0_1_" + STARTTIME + ".png")

        r_0_9_predict = regressor_0_9.predict(X_test)

        plot_predictions(
            yscaler.inverse_transform(r_0_9_predict),
            yscaler.inverse_transform(y_test),
            df_test["color_cleaned"].values,
            outname="quantile_0_9_" + STARTTIME + ".png",
        )
        experiment.log_image("quantile_0_9_" + STARTTIME + ".png")


if __name__ == "__main__":
    main()
