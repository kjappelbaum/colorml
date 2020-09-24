# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import pickle
import time

import click
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from six.moves import range
from sklearn.ensemble import BaggingRegressor
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from xgboost import XGBRegressor

import wandb

from ..utils.descriptornames import *
from ..utils.utils import (augment_data, pairwise_delta_es, plot_predictions, read_pickle)

with open(
        '/Users/kevinmaikjablonka/Dropbox (LSMO)/proj75_mofcolor/ml/data/color_threshold.pkl',
        'rb',
) as fh:
    color_threshold_dict = pickle.load(fh)

# upbeat-sweep-85 #volcanic-sweep-182 #proud-sweep-227
PARAMETERS_MEDIAN = {
    'colsample_bytree': 0.5341,
    'max_depth': 33,
    'min_child_weight': 0.001713,
    'n_estimators': 3650,
    'num_leaves': 16,
    'reg_alpha': 1.361,
    'reg_lambda': 1.484,
    'subsample': 0.2767,
}

# amber-sweep-132 #dark-sweep-916
PARAMETERS_01 = {
    'colsample_bytree': 0.04045,
    'max_depth': 22,
    'min_child_weight': 0.008584,
    'n_estimators': 16,
    'num_leaves': 268,
    'reg_alpha': 1.279,
    'reg_lambda': 1.349,
    'subsample': 0.8936,
}

# driven-sweep-41
PARAMETERS_09 = {
    'colsample_bytree': 0.3349,
    'max_depth': 44,
    'min_child_weight': 0.06702,
    'n_estimators': 4875,
    'num_leaves': 285,
    'reg_alpha': 1.107,
    'reg_lambda': 1.051,
    'subsample': 0.9687,
}

RANDOM_SEED = int(20021994)

CHEMICAL_FEATURES = [
    'mc_CRY-chi-0-all',
    'mc_CRY-chi-1-all',
    'mc_CRY-chi-2-all',
    'mc_CRY-chi-3-all',
    'mc_CRY-Z-0-all',
    'mc_CRY-Z-1-all',
    'mc_CRY-Z-2-all',
    'mc_CRY-Z-3-all',
    'mc_CRY-I-1-all',
    'mc_CRY-I-2-all',
    'mc_CRY-I-3-all',
    'mc_CRY-T-0-all',
    'mc_CRY-T-1-all',
    'mc_CRY-T-2-all',
    'mc_CRY-T-3-all',
    'mc_CRY-S-0-all',
    'mc_CRY-S-1-all',
    'mc_CRY-S-2-all',
    'mc_CRY-S-3-all',
    'D_mc_CRY-chi-1-all',
    'D_mc_CRY-chi-2-all',
    'D_mc_CRY-chi-3-all',
    'D_mc_CRY-Z-1-all',
    'D_mc_CRY-Z-2-all',
    'D_mc_CRY-Z-3-all',
    'D_mc_CRY-T-1-all',
    'D_mc_CRY-T-2-all',
    'D_mc_CRY-T-3-all',
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
    'func-I-1-all',
    'func-I-2-all',
    'func-I-3-all',
    'func-T-0-all',
    'func-T-1-all',
    'func-T-2-all',
    'func-T-3-all',
    'func-S-0-all',
    'func-S-1-all',
    'func-S-2-all',
    'func-S-3-all',
    'func-alpha-0-all',
    'func-alpha-1-all',
    'func-alpha-2-all',
    'func-alpha-3-all',
    'D_func-chi-1-all',
    'D_func-chi-2-all',
    'D_func-chi-3-all',
    'D_func-Z-1-all',
    'D_func-Z-2-all',
    'D_func-Z-3-all',
    'D_func-T-1-all',
    'D_func-T-2-all',
    'D_func-T-3-all',
    'D_func-S-2-all',
    'D_func-S-3-all',
    'D_func-alpha-1-all',
    'D_func-alpha-2-all',
    'D_func-alpha-3-all',
    'f-lig-chi-0',
    'f-lig-chi-1',
    'f-lig-chi-2',
    'f-lig-chi-3',
    'f-lig-Z-0',
    'f-lig-Z-1',
    'f-lig-Z-2',
    'f-lig-Z-3',
    'f-lig-I-0',
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
    'lc-I-2-all',
    'lc-I-3-all',
    'lc-T-0-all',
    'lc-T-1-all',
    'lc-T-2-all',
    'lc-T-3-all',
    'lc-S-3-all',
    'lc-alpha-0-all',
    'lc-alpha-1-all',
    'lc-alpha-2-all',
    'lc-alpha-3-all',
    'D_lc-chi-2-all',
    'D_lc-chi-3-all',
    'D_lc-Z-1-all',
    'D_lc-Z-2-all',
    'D_lc-Z-3-all',
    'D_lc-T-1-all',
    'D_lc-T-2-all',
    'D_lc-T-3-all',
    'D_lc-alpha-1-all',
    'D_lc-alpha-2-all',
    'D_lc-alpha-3-all',
    'tertiary_amide_sum',
    'ester_sum',
    'carbonyl_sum',
    'logP_sum',
    'MR_sum',
    'aromatic_rings_sum',
    'dbonds_sum',
    'abonds_sum',
    'tertiary_amide_mean',
    'ester_mean',
    'carbonyl_mean',
    'logP_mean',
    'MR_mean',
    'aromatic_rings_mean',
    'dbonds_mean',
    'abonds_mean',
    # "sum-func-chi-0-all",
    # "sum-func-chi-1-all",
    # "sum-func-chi-2-all",
    # "sum-func-chi-3-all",
    # "sum-func-Z-0-all",
    # "sum-func-Z-1-all",
    # "sum-func-Z-2-all",
    # "sum-func-Z-3-all",
    # "sum-func-I-0-all",
    # "sum-func-I-1-all",
    # "sum-func-I-2-all",
    # "sum-func-I-3-all",
    # "sum-func-T-0-all",
    # "sum-func-T-1-all",
    # "sum-func-T-2-all",
    # "sum-func-T-3-all",
    # "sum-func-S-0-all",
    # "sum-func-S-1-all",
    # "sum-func-S-2-all",
    # "sum-func-S-3-all",
    # "sum-func-alpha-0-all",
    # "sum-func-alpha-1-all",
    # "sum-func-alpha-2-all",
    # "sum-func-alpha-3-all",
    # "sum-D_func-chi-1-all",
    # "sum-D_func-chi-2-all",
    # "sum-D_func-chi-3-all",
    # "sum-D_func-Z-1-all",
    # "sum-D_func-Z-2-all",
    # "sum-D_func-Z-3-all",
    # "sum-D_func-T-1-all",
    # "sum-D_func-T-2-all",
    # "sum-D_func-T-3-all",
    # "sum-D_func-S-1-all",
    # "sum-D_func-S-2-all",
    # "sum-D_func-S-3-all",
    # "sum-D_func-alpha-1-all",
    # "sum-D_func-alpha-2-all",
    # "sum-D_func-alpha-3-all",
    # "sum-f-lig-chi-0",
    # "sum-f-lig-chi-1",
    # "sum-f-lig-chi-2",
    # "sum-f-lig-chi-3",
    # "sum-f-lig-Z-0",
    # "sum-f-lig-Z-1",
    # "sum-f-lig-Z-2",
    # "sum-f-lig-Z-3",
    # "sum-f-lig-I-0",
    # "sum-f-lig-I-1",
    # "sum-f-lig-I-2",
    # "sum-f-lig-I-3",
    # "sum-f-lig-T-0",
    # "sum-f-lig-T-1",
    # "sum-f-lig-T-2",
    # "sum-f-lig-T-3",
    # "sum-f-lig-S-0",
    # "sum-f-lig-S-1",
    # "sum-f-lig-S-2",
    # "sum-f-lig-S-3",
    # "sum-lc-chi-0-all",
    # "sum-lc-chi-1-all",
    # "sum-lc-chi-2-all",
    # "sum-lc-chi-3-all",
    # "sum-lc-Z-0-all",
    # "sum-lc-Z-1-all",
    # "sum-lc-Z-2-all",
    # "sum-lc-Z-3-all",
    # "sum-lc-I-0-all",
    # "sum-lc-I-1-all",
    # "sum-lc-I-2-all",
    # "sum-lc-I-3-all",
    # "sum-lc-T-0-all",
    # "sum-lc-T-1-all",
    # "sum-lc-T-2-all",
    # "sum-lc-T-3-all",
    # "sum-lc-S-0-all",
    # "sum-lc-S-1-all",
    # "sum-lc-S-2-all",
    # "sum-lc-S-3-all",
    # "sum-lc-alpha-0-all",
    # "sum-lc-alpha-1-all",
    # "sum-lc-alpha-2-all",
    # "sum-lc-alpha-3-all",
    # "sum-D_lc-chi-1-all",
    # "sum-D_lc-chi-2-all",
    # "sum-D_lc-chi-3-all",
    # "sum-D_lc-Z-1-all",
    # "sum-D_lc-Z-2-all",
    # "sum-D_lc-Z-3-all",
    # "sum-D_lc-T-1-all",
    # "sum-D_lc-T-2-all",
    # "sum-D_lc-T-3-all",
    # "sum-D_lc-S-1-all",
    # "sum-D_lc-S-2-all",
    # "sum-D_lc-S-3-all",
    # "sum-D_lc-alpha-1-all",
    # "sum-D_lc-alpha-2-all",
    # "sum-D_lc-alpha-3-all",
    # "sum-mc_CRY-chi-0-all",
    # "sum-mc_CRY-chi-1-all",
    # "sum-mc_CRY-chi-2-all",
    # "sum-mc_CRY-chi-3-all",
    # "sum-mc_CRY-Z-0-all",
    # "sum-mc_CRY-Z-1-all",
    # "sum-mc_CRY-Z-2-all",
    # "sum-mc_CRY-Z-3-all",
    # "sum-mc_CRY-I-0-all",
    # "sum-mc_CRY-I-1-all",
    # "sum-mc_CRY-I-2-all",
    # "sum-mc_CRY-I-3-all",
    # "sum-mc_CRY-T-0-all",
    # "sum-mc_CRY-T-1-all",
    # "sum-mc_CRY-T-2-all",
    # "sum-mc_CRY-T-3-all",
    # "sum-mc_CRY-S-0-all",
    # "sum-mc_CRY-S-1-all",
    # "sum-mc_CRY-S-2-all",
    # "sum-mc_CRY-S-3-all",
    # "sum-D_mc_CRY-chi-1-all",
    # "sum-D_mc_CRY-chi-2-all",
    # "sum-D_mc_CRY-chi-3-all",
    # "sum-D_mc_CRY-Z-1-all",
    # "sum-D_mc_CRY-Z-2-all",
    # "sum-D_mc_CRY-Z-3-all",
    # "sum-D_mc_CRY-T-1-all",
    # "sum-D_mc_CRY-T-2-all",
    # "sum-D_mc_CRY-T-3-all",
    # "sum-D_mc_CRY-S-1-all",
    # "sum-D_mc_CRY-S-2-all",
    # "sum-D_mc_CRY-S-3-all",
]

AUGMENT_DICT = ('/Users/kevinmaikjablonka/Dropbox (LSMO)/proj75_mofcolor/ml/data/augment_dict.pkl')


def split_data(threshold):
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    df = pd.read_csv('/Users/kevinmaikjablonka/Dropbox (LSMO)/proj75_mofcolor/ml/data/all.csv')
    exlcuded = []
    keept = []

    THRESHOLD = 0.03

    if threshold != 255:
        for i, row in df.iterrows():
            if row['color_cleaned_x'] in color_threshold_dict[threshold]:
                keept.append(row)
            else:
                exlcuded.append(row)
        df_rel = pd.DataFrame(keept)
        df = df_rel.drop_duplicates(subset=CHEMICAL_FEATURES)
    else:
        df = df.drop_duplicates(subset=CHEMICAL_FEATURES)
    r_binned = bin_column(df['r'].values)
    g_binned = bin_column(df['g'].values)
    b_binned = bin_column(df['b'].values)
    mlss = MultilabelStratifiedShuffleSplit(n_splits=1, train_size=0.85, test_size=0.15, random_state=RANDOM_SEED)
    for train_idx, test_idx in mlss.split(df, np.hstack([r_binned, g_binned, b_binned])):
        pass
    df_train = df.iloc[train_idx].sample(len(train_idx))
    df_test = df.iloc[test_idx].sample(len(test_idx))
    df_train.to_csv(
        '/Users/kevinmaikjablonka/Dropbox (LSMO)/proj75_mofcolor/ml/data/development_set.csv',
        index=False,
    )
    df_test.to_csv(
        '/Users/kevinmaikjablonka/Dropbox (LSMO)/proj75_mofcolor/ml/data/holdout_set.csv',
        index=False,
    )


def bin_column(column):
    binned = []
    for value in column:
        if value < 85:
            binned.append(0)
        elif 85 <= value < 170:
            binned.append(1)
        else:
            binned.append(2)
    return np.array(binned).reshape(-1, 1)


def process_data(augment=False):
    df_train = pd.read_csv('/Users/kevinmaikjablonka/Dropbox (LSMO)/proj75_mofcolor/ml/data/development_set.csv')
    df_test = pd.read_csv('/Users/kevinmaikjablonka/Dropbox (LSMO)/proj75_mofcolor/ml/data/holdout_set.csv')

    if augment:
        augment_dict = read_pickle(AUGMENT_DICT)
        df_train = augment_data(df_train, augment_dict, name_col='color_cleaned_x')

    X_train = df_train[CHEMICAL_FEATURES]
    X_test = df_test[CHEMICAL_FEATURES]

    y_train = df_train[['r', 'g', 'b']].values / 255
    y_test = df_test[['r', 'g', 'b']].values / 255

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return scaler, None, X_train, X_test, y_train, y_test, df_test


def fit(x_train, y_train, parameters_01, parameters_median, parameters_09):
    regressor_median = BaggingRegressor(MultiOutputRegressor(
        LGBMRegressor(objective='quantile', alpha=0.5, **parameters_median)),
                                        n_jobs=-1,
                                        n_estimators=15)
    regressor_median.fit(x_train, y_train)

    regressor_0_1 = MultiOutputRegressor(LGBMRegressor(objective='quantile', alpha=0.1, **parameters_01))
    regressor_0_1.fit(x_train, y_train)

    regressor_0_9 = MultiOutputRegressor(LGBMRegressor(objective='quantile', alpha=0.9, **parameters_09))

    regressor_0_9.fit(x_train, y_train)

    return regressor_median, regressor_0_1, regressor_0_9


@click.command('cli')
@click.option('--augment', is_flag=True)
@click.argument('repeats', type=int, default=1)
@click.argument('threshold', type=int, default=16)
def main(augment, repeats, threshold):
    delta_es = []
    # delta_es_test = []
    # df_test_all = pd.read_csv("/Users/kevinmaikjablonka/Dropbox (LSMO)/proj75_mofcolor/ml/data/to_sample_from_test.csv")
    # X_test_all = df_test_all[CHEMICAL_FEATURES]

    # y_test_all = df_test_all[["r", "g", "b"]].values / 255

    for i in range(repeats):
        STARTTIME = time.strftime('run_%Y_%m_%d_%H_%M_%s')
        split_data(threshold)
        scaler, yscaler, X_train, X_test, y_train, y_test, df_test = process_data(augment)

        joblib.dump(scaler, 'scaler_' + STARTTIME + '.joblib')

        np.save('X_train_' + STARTTIME + '.npy', X_train)
        np.save('X_test_' + STARTTIME + '.npy', X_test)
        np.save('y_train_' + STARTTIME + '.npy', y_train)
        np.save('y_test_' + STARTTIME + '.npy', y_test)
        np.save('y_names_' + STARTTIME + '.npy', df_test['color_cleaned_x'])

        wandb.save('scaler_' + STARTTIME + '.joblib')

        wandb.save('X_train_' + STARTTIME + '.npy')
        wandb.save('X_test_' + STARTTIME + '.npy')
        wandb.save('y_train_' + STARTTIME + '.npy')
        wandb.save('y_test_' + STARTTIME + '.npy')
        wandb.save('y_names_' + STARTTIME + '.npy')

        regressor_median, regressor_0_1, regressor_0_9 = fit(X_train, y_train, PARAMETERS_01, PARAMETERS_MEDIAN,
                                                             PARAMETERS_09)

        joblib.dump(regressor_median, 'regressor_median' + STARTTIME + str(augment) + '.joblib', compress=3)
        joblib.dump(regressor_0_1, 'regressor_0_1' + STARTTIME + str(augment) + '.joblib', compress=3)
        joblib.dump(regressor_0_9, 'regressor_0_9' + STARTTIME + str(augment) + '.joblib', compress=3)

        # wandb.save("regressor_median" + STARTTIME + str(augment) + ".joblib")
        # wandb.save("regressor_0_1" + STARTTIME + str(augment) + ".joblib")
        # wandb.save("regressor_0_9" + STARTTIME + str(augment) + ".joblib")

        median_predict = regressor_median.predict(X_test)

        plot_predictions(
            median_predict * 255,
            y_test * 255,
            df_test['color_cleaned_x'].values,
            outname='median_' + STARTTIME + str(augment) + '.png',
        )

        differences = pairwise_delta_es(y_test, median_predict)
        delta_es.append(differences)

        # median_predict = regressor_median.predict(X_test_all)

        # plot_predictions(
        #     median_predict * 255,
        #     y_test * 255,
        #     df_test["color_cleaned_x"].values,
        #     outname="median_" + STARTTIME + str(augment) + ".png",
        # )

        # differences = pairwise_delta_es(y_test_all, median_predict)
        # delta_es_test.append(differences)

        # wandb.log({"example": wandb.Image("median_" + STARTTIME + str(augment) + ".png")})

        r_0_1_predict = regressor_0_1.predict(X_test)

        plot_predictions(
            r_0_1_predict * 255,
            y_test * 255,
            df_test['color_cleaned_x'].values,
            outname='quantile_0_1_' + STARTTIME + str(augment) + '.png',
        )
        wandb.log({'example': wandb.Image('quantile_0_1_' + STARTTIME + str(augment) + '.png')})

        r_0_9_predict = regressor_0_9.predict(X_test)

        plot_predictions(
            r_0_9_predict * 255,
            y_test * 255,
            df_test['color_cleaned_x'].values,
            outname='quantile_0_9_' + STARTTIME + str(augment) + '.png',
        )

        wandb.log({'example': wandb.Image('quantile_0_9_' + STARTTIME + str(augment) + '.png')})

    np.save('delta_e_{}_{}'.format(STARTTIME, str(augment)), delta_es)
    # np.save("delta_e_{}_{}_test".format(STARTTIME, str(augment)), delta_es_test)


if __name__ == '__main__':
    wandb.init(project='colorml')
    main()
