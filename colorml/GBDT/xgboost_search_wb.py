# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import pickle
from functools import partial

import click
import numpy as np
import pandas as pd
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor, sRGBColor
from colour.models import (RGB_to_HSL, RGB_to_HSV, RGB_to_XYZ, XYZ_to_Lab, XYZ_to_RGB, XYZ_to_xy)
from colour.plotting import filter_RGB_colourspaces
from colour.utilities import first_item
from lightgbm import LGBMRegressor
from six.moves import zip
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from . import wandb

# from ..utils.descriptornames import *  # pylint:disable=unused-wildcard-import

colourspace = first_item(filter_RGB_colourspaces('sRGB').values())


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

config = {
    'n_estimators': {
        'distribution': 'int_uniform',
        'min': 10,
        'max': 5000
    },
    'max_depth': {
        'distribution': 'int_uniform',
        'min': 5,
        'max': 100
    },
    'num_leaves': {
        'distribution': 'int_uniform',
        'min': 5,
        'max': 500
    },
    'reg_alpha': {
        'distribution': 'log_uniform',
        'min': 0.00001,
        'max': 0.4
    },
    'reg_lambda': {
        'distribution': 'log_uniform',
        'min': 0.00001,
        'max': 0.4
    },
    'subsample': {
        'distribution': 'uniform',
        'min': 0.01,
        'max': 1.0
    },
    'colsample_bytree': {
        'distribution': 'uniform',
        'min': 0.01,
        'max': 1.0
    },
    'min_child_weight': {
        'distribution': 'uniform',
        'min': 0.001,
        'max': 0.1,
    },
}

CHEMICAL_FEATURES = [
    'f-lig-chi-0',
    'f-lig-chi-1',
    'f-lig-chi-2',
    'f-lig-chi-3',
    'f-lig-Z-0',
    'f-lig-Z-1',
    'f-lig-Z-2',
    'f-lig-Z-3',
    'f-lig-I-1',
    'f-lig-I-2',
    'f-lig-I-3',
    'f-lig-T-0',
    'f-lig-T-1',
    'f-lig-T-2',
    'f-lig-T-3',
    'f-lig-S-0',
    'f-lig-S-1',
    'f-lig-S-2',
    'f-lig-S-3',
    'lc-chi-0-all',
    'lc-chi-1-all',
    'lc-chi-2-all',
    'lc-chi-3-all',
    'lc-Z-0-all',
    'lc-Z-1-all',
    'lc-Z-2-all',
    'lc-Z-3-all',
    'lc-I-0-all',
    'lc-I-1-all',
    'lc-I-2-all',
    'lc-I-3-all',
    'lc-T-0-all',
    'lc-T-2-all',
    'lc-S-1-all',
    'lc-S-2-all',
    'lc-alpha-2-all',
    'lc-alpha-3-all',
    'D_lc-chi-0-all',
    'D_lc-chi-1-all',
    'D_lc-Z-1-all',
    'D_lc-I-1-all',
    'D_lc-I-2-all',
    'D_lc-I-3-all',
    'D_lc-T-0-all',
    'D_lc-T-1-all',
    'D_lc-T-2-all',
    'D_lc-T-3-all',
    'D_lc-S-0-all',
    'D_lc-S-1-all',
    'D_lc-S-2-all',
    'D_lc-S-3-all',
    'D_lc-alpha-0-all',
    'D_lc-alpha-1-all',
    'D_lc-alpha-2-all',
    'D_lc-alpha-3-all',
    'mc_CRY-chi-0-all',
    'mc_CRY-chi-1-all',
    'mc_CRY-chi-2-all',
    'mc_CRY-Z-0-all',
    'mc_CRY-Z-1-all',
    'mc_CRY-Z-2-all',
    'mc_CRY-Z-3-all',
    'mc_CRY-I-3-all',
    'mc_CRY-T-1-all',
    'mc_CRY-T-2-all',
    'mc_CRY-T-3-all',
    'mc_CRY-S-3-all',
    'D_mc_CRY-chi-0-all',
    'D_mc_CRY-chi-1-all',
    'D_mc_CRY-chi-2-all',
    'D_mc_CRY-chi-3-all',
    'D_mc_CRY-Z-1-all',
    'D_mc_CRY-Z-2-all',
    'D_mc_CRY-Z-3-all',
    'D_mc_CRY-I-1-all',
    'D_mc_CRY-I-3-all',
    'D_mc_CRY-T-0-all',
    'D_mc_CRY-T-1-all',
    'D_mc_CRY-T-2-all',
    'D_mc_CRY-S-0-all',
    'D_mc_CRY-S-1-all',
    'D_mc_CRY-S-2-all',
    'D_mc_CRY-S-3-all',
    'func-chi-0-all',
    'func-chi-1-all',
    'func-chi-2-all',
    'func-chi-3-all',
    'func-Z-0-all',
    'func-Z-1-all',
    'func-Z-2-all',
    'func-Z-3-all',
    'func-I-0-all',
    'func-I-1-all',
    'func-T-1-all',
    'func-T-2-all',
    'func-S-0-all',
    'func-S-1-all',
    'primary_amide_sum',
    'secondary_amide_sum',
    'tertiary_amide_sum',
    'ester_sum',
    'carbonyl_sum',
    'logP_sum',
    'MR_sum',
    'dbratio_sum',
    'aromatic_rings_sum',
    'dbonds_sum',
    'abonds_sum',
]

df = pd.read_csv('development_set.csv')

exlcuded = []
keept = []

with open(
        'olor_threshold.pkl',
        'rb',
) as fh:
    color_threshold_dict = pickle.load(fh)

for i, row in df.iterrows():
    if row['color_cleaned_x'] in color_threshold_dict[16]:
        keept.append(row)
    else:
        exlcuded.append(row)
df_rel = pd.DataFrame(keept)
df = df_rel.drop_duplicates(subset=CHEMICAL_FEATURES)

X_train = df[CHEMICAL_FEATURES]
X_test = df[CHEMICAL_FEATURES]

y_train = df[['r', 'g', 'b']].values / 255
y_test = df[['r', 'g', 'b']].values / 255

# Based on the feature selection we now have, there is no need for a VT
# vt = VarianceThreshold(0.2)  # remove the constant features

# X_train = vt.fit_transform(X_train)
# X_test = vt.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


# return us a sweep id (required for running the sweep)
def get_sweep_id(method):
    sweep_config = {
        'method': method,
        'metric': {
            'name': 'cv_mean',
            'goal': 'minimize'
        },
        'early_terminate': {
            'type': 'hyperband',
            's': 2,
            'eta': 3,
            'max_iter': 30
        },
        'parameters': config,
    }
    sweep_id = wandb.sweep(sweep_config, project='colorml')

    return sweep_id


def train(alpha=0.5, delta_e_loss=True):
    # Config is a variable that holds and saves hyperparameters and inputs

    configs = {
        'n_estimators': 100,
        'max_depth': 10,
        'num_leaves': 50,
        'reg_alpha': 0.00001,
        'reg_lambda': 0.00001,
        'subsample': 0.2,
        'colsample_bytree': 0.2,
        'min_child_weight': 0.001,
    }

    # Initilize a new wandb run
    wandb.init(project='colorml', config=configs)

    config = wandb.config

    regressor = MultiOutputRegressor(LGBMRegressor(objective='quantile', alpha=alpha, **config))

    if delta_e_loss:
        cv = cross_val_score(regressor, X_train, y_train, n_jobs=5, scoring=scorer, cv=5)
    else:
        cv = cross_val_score(regressor, X_train, y_train, n_jobs=2, cv=5)

    mean = np.abs(cv.mean())
    std = np.abs(cv.std())
    wandb.log({'cv_mean': mean})
    wandb.log({'cv_std': std})

    wandb.run.summary['cv_mean'] = mean
    wandb.run.summary['cv_std'] = std


@click.command('cli')
@click.argument('alpha')
@click.option('--delta_e_loss', is_flag=True)
def main(alpha, delta_e_loss):

    print((alpha, delta_e_loss))
    train_func = partial(train, alpha=alpha, delta_e_loss=delta_e_loss)
    sweep_id = get_sweep_id('bayes')

    wandb.agent(sweep_id, function=train_func)


if __name__ == '__main__':
    main()
