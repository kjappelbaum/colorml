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

from .descriptornames import *
from .utils import augment_data, plot_predictions, read_pickle

PARAMETERS = {
    'colsample_bytree': 0.8812456128806658,
    'max_depth': 28,
    'min_child_weight': 0.04523243980083686,
    'n_estimators': 3609,
    'num_leaves': 370,
    'reg_alpha': 0.00014309470282004296,
    'reg_lambda': 0.00015498366357705995,
    'subsample': 0.8611031439581149,
}

RANDOM_SEED = int(84996)

CHEMICAL_FEATURES = (metalcenter_descriptors + functionalgroup_descriptors + linker_descriptors + mol_desc +
                     summed_functionalgroup_descriptors + summed_linker_descriptors + summed_metalcenter_descriptors)

AUGMENT_DICT = '../data/augment_dict.pkl'


def process_data(augment=False):
    df = pd.read_csv('../data/color_feat_merged.csv')
    df_train, df_test = train_test_split(df, train_size=0.8, random_state=RANDOM_SEED)

    if augment:
        augment_dict = read_pickle(AUGMENT_DICT)
        df_train = augment_data(df_train, augment_dict)

    X_train = df_train[CHEMICAL_FEATURES]
    X_test = df_test[CHEMICAL_FEATURES]

    yscaler = StandardScaler()
    y_train = yscaler.fit_transform(df_train[['r', 'g', 'b']])
    y_test = yscaler.transform(df_test[['r', 'g', 'b']])
    vt = VarianceThreshold(0.3)  # remove the constant features

    X_train = vt.fit_transform(X_train)
    X_test = vt.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return vt, scaler, yscaler, X_train, X_test, y_train, y_test, df_test


def fit(x_train, y_train, parameters):
    regressor_mean = MultiOutputRegressor(LGBMRegressor(**parameters))
    regressor_mean.fit(x_train, y_train)

    regressor_median = MultiOutputRegressor(LGBMRegressor(objective='quantile', alpha=0.5, **parameters))
    regressor_median.fit(x_train, y_train)

    regressor_0_1 = MultiOutputRegressor(LGBMRegressor(objective='quantile', alpha=0.1, **parameters))
    regressor_0_1.fit(x_train, y_train)

    regressor_0_9 = MultiOutputRegressor(LGBMRegressor(objective='quantile', alpha=0.9, **parameters))
    regressor_0_9.fit(x_train, y_train)

    return regressor_mean, regressor_median, regressor_0_1, regressor_0_9


def main():
    STARTTIME = time.strftime('run_%Y_%m_%d_%H_%M_%s')
    vt, scaler, yscaler, X_train, X_test, y_train, y_test, df_test = process_data()

    joblib.dump(scaler, 'scaler_' + STARTTIME + '.joblib')
    joblib.dump(vt, 'vt_' + STARTTIME + '.joblib')
    joblib.dump(yscaler, 'yscaler_' + STARTTIME + '.joblib')
    np.save('X_train_' + STARTTIME + '.npy', X_train)
    np.save('X_test_' + STARTTIME + '.npy', X_test)
    np.save('y_train_' + STARTTIME + '.npy', y_train)
    np.save('y_test_' + STARTTIME + '.npy', y_test)
    np.save('y_names_' + STARTTIME + '.npy', df_test['color_cleaned'])

    experiment = Experiment(api_key=os.environ['COMET_API_KEY'], project_name='color-ml')

    experiment.log_asset('scaler_' + STARTTIME + '.joblib')
    experiment.log_asset('vt_' + STARTTIME + '.joblib')
    experiment.log_asset('yscaler_' + STARTTIME + '.joblib')
    experiment.log_asset('X_train_' + STARTTIME + '.npy')
    experiment.log_asset('X_test_' + STARTTIME + '.npy')
    experiment.log_asset('y_train_' + STARTTIME + '.npy')
    experiment.log_asset('y_test_' + STARTTIME + '.npy')
    experiment.log_asset('y_names_' + STARTTIME + '.npy')

    experiment.log_parameters(PARAMETERS)

    with experiment.train():
        regressor_mean, regressor_median, regressor_0_1, regressor_0_9 = fit(X_train, y_train, PARAMETERS)

    joblib.dump(regressor_mean, 'regressor_mean' + STARTTIME + '.joblib')
    joblib.dump(regressor_median, 'regressor_median' + STARTTIME + '.joblib')
    joblib.dump(regressor_0_1, 'regressor_0_1' + STARTTIME + '.joblib')
    joblib.dump(regressor_0_9, 'regressor_0_9' + STARTTIME + '.joblib')

    experiment.log_asset('regressor_mean' + STARTTIME + '.joblib')
    experiment.log_asset('regressor_median' + STARTTIME + '.joblib')
    experiment.log_asset('regressor_0_1' + STARTTIME + '.joblib')
    experiment.log_asset('regressor_0_9' + STARTTIME + '.joblib')

    with experiment.test():
        mean_predict = regressor_mean.predict(X_test)
        r2 = r2_score(y_test, mean_predict)
        mae = mean_absolute_error(y_test, mean_predict)
        mse = mean_squared_error(y_test, mean_predict)

        experiment.log_metric('r2_score', r2)
        experiment.log_metric('mae', mae)
        experiment.log_metric('mse', mse)

        plot_predictions(
            yscaler.inverse_transform(mean_predict),
            yscaler.inverse_transform(y_test),
            df_test['color_cleaned'].values,
            outname='mean_' + STARTTIME + '.png',
        )
        experiment.log_image('mean_' + STARTTIME + '.png')

        median_predict = regressor_median.predict(X_test)

        plot_predictions(
            yscaler.inverse_transform(median_predict),
            yscaler.inverse_transform(y_test),
            df_test['color_cleaned'].values,
            outname='median_' + STARTTIME + '.png',
        )

        experiment.log_image('median_' + STARTTIME + '.png')

        r_0_1_predict = regressor_0_1.predict(X_test)

        plot_predictions(
            yscaler.inverse_transform(r_0_1_predict),
            yscaler.inverse_transform(y_test),
            df_test['color_cleaned'].values,
            outname='quantile_0_1_' + STARTTIME + '.png',
        )
        experiment.log_image('quantile_0_1_' + STARTTIME + '.png')

        r_0_9_predict = regressor_0_9.predict(X_test)

        plot_predictions(
            yscaler.inverse_transform(r_0_9_predict),
            yscaler.inverse_transform(y_test),
            df_test['color_cleaned'].values,
            outname='quantile_0_9_' + STARTTIME + '.png',
        )
        experiment.log_image('quantile_0_9_' + STARTTIME + '.png')


if __name__ == '__main__':
    main()
