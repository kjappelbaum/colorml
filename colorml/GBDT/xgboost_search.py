# -*- coding: utf-8 -*-
"""Use the cometml optimizer to search over the GBDT parameterspace"""
from __future__ import absolute_import

import os

import click
import numpy as np
import pandas as pd
from comet_ml import Optimizer
from colour.models import (
    RGB_to_HSL,
    RGB_to_HSV,
    RGB_to_XYZ,
    XYZ_to_Lab,
    XYZ_to_RGB,
    XYZ_to_xy,
)
from colour.plotting import filter_RGB_colourspaces
from colour.utilities import first_item

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from ..utils.descriptornames import *  # pylint:disable=unused-wildcard-import

colourspace = first_item(filter_RGB_colourspaces("sRGB").values())

from sklearn.metrics.scorer import make_scorer


def get_delta_e(rgba, rgbb):
    color1_rgb = sRGBColor(rgba[0], rgba[1], rgba[2])
    color2_rgb = sRGBColor(rgbb[0], rgbb[1], rgbb[2])

    # Convert from RGB to Lab Color Space
    color1_lab = convert_color(color1_rgb, LabColor)

    # Convert from RGB to Lab Color Space
    color2_lab = convert_color(color2_rgb, LabColor)

    # Find the color difference
    delta_e = delta_e_cie2000(color1_lab, color2_lab)

    return delta_e


def delta_e_loss_from_rgb(true, prediction):
    distances = []
    for t, p in zip(true, prediction):
        distances.append(get_delta_e(t, p))
    return np.array(distances).mean()


scorer = make_scorer(delta_e_loss_from_rgb, greater_is_better=False)

RANDOM_SEED = int(821996)

# CHEMICAL_FEATURES = (
#     metalcenter_descriptors
#     + functionalgroup_descriptors
#     + linker_descriptors
#     + mol_desc
#     + summed_functionalgroup_descriptors
#     + summed_linker_descriptors
#     + summed_metalcenter_descriptors
# )

# The selection below comes from an initial LightGBM model, where we simply drop the 150 least important features.
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


def process_data():
    df = pd.read_csv("development_set.csv")
    df_train, df_test = train_test_split(df, train_size=0.8, random_state=RANDOM_SEED)
    df_train = df_train.sample(3000)
    X_train = df_train[CHEMICAL_FEATURES]
    X_test = df_test[CHEMICAL_FEATURES]

    y_train = df_train[["r", "g", "b"]].values / 255
    y_test = df_test[["r", "g", "b"]].values / 255

    # Based on the feature selection we now have, there is no need for a VT
    # vt = VarianceThreshold(0.2)  # remove the constant features

    # X_train = vt.fit_transform(X_train)
    # X_test = vt.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return scaler, None, X_train, X_test, y_train, y_test


def fit(
    experiment,
    x_train,
    y_train,
    parameters,
    alpha: float = 0.5,
    delta_e_loss: bool = True,
):
    regressor = MultiOutputRegressor(
        LGBMRegressor(objective="quantile", alpha=alpha, **parameters)
    )

    if delta_e_loss:
        cv = cross_val_score(regressor, x_train, y_train, n_jobs=-1, scoring=scorer)
    else:
        cv = cross_val_score(regressor, x_train, y_train, n_jobs=-1)

    return np.abs(cv.mean()), cv.std()


def main(alpha=0.5):
    scaler, yscaler, X_train, X_test, y_train, y_test = process_data()

    config = {
        # We pick the Bayes algorithm:
        "algorithm": "bayes",
        "name": "fine tune LightGBM",
        # Declare your hyperparameters in the Vizier-inspired format:
        "parameters": {
            "n_estimators": {"type": "integer", "min": 50, "max": 5000},
            "max_depth": {"type": "integer", "min": 10, "max": 50},
            "num_leaves": {"type": "integer", "min": 100, "max": 500},
            "reg_alpha": {
                "type": "float",
                "min": 0.00001,
                "max": 0.2,
                "scalingType": "loguniform",
            },
            "reg_lambda": {
                "type": "float",
                "min": 0.00001,
                "max": 0.2,
                "scalingType": "loguniform",
            },
            "subsample": {"type": "float", "min": 0.2, "max": 1.0},
            "colsample_bytree": {"type": "float", "min": 0.2, "max": 1.0},
            "min_child_weight": {
                "type": "float",
                "min": 0.001,
                "max": 0.1,
                "scalingType": "loguniform",
            },
        },
        # Declare what we will be optimizing, and how:
        "spec": {"metric": "loss", "objective": "minimize"},
    }

    # Next, create an optimizer, passing in the config:
    # (You can leave out API_KEY if you already set it)
    opt = Optimizer(
        config, api_key=os.environ["COMET_API_KEY"], project_name="color-ml"
    )

    for _, experiment in enumerate(opt.get_experiments()):
        experiment.log_parameter("colorspace", "rgb")
        params = {
            "n_estimators": experiment.get_parameter("n_estimators"),
            "colsample_bytree": experiment.get_parameter("colsample_bytree"),
            "num_leaves": experiment.get_parameter("num_leaves"),
            "max_depth": experiment.get_parameter("max_depth"),
            "reg_alpha": experiment.get_parameter("reg_alpha"),
            "reg_lambda": experiment.get_parameter("reg_lambda"),
            "subsample": experiment.get_parameter("subsample"),
            "min_child_weight": experiment.get_parameter("min_child_weight"),
        }
        loss, std = fit(experiment, X_train, y_train, params, alpha)
        experiment.log_metric("loss", loss)
        experiment.log_metric("std", std)


if __name__ == "__main__":
    main()
