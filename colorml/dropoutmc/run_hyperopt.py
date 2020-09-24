# -*- coding: utf-8 -*-
# pylint:disable=unused-import
"""Use hyperopt to optimize the MLP with Dropout MC"""
from __future__ import absolute_import, print_function

import sys

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from comet_ml import Experiment
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from keras import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.constraints import MinMaxNorm
from keras.datasets import mnist
from keras.initializers import Constant
from keras.layers import (Activation, Dense, Dropout, GaussianDropout, GaussianNoise, Input, LeakyReLU, concatenate)
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l1
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ..utils.descriptornames import *  # pylint:disable=unused-wildcard-import
from ..utils.utils import (augment_data, get_timestamp_string, huber_fn, mapping_to_target_range,
                           mapping_to_target_range_sig, plot_predictions, read_pickle)

lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)

sys.path.append('../')


def data():
    CHEMICAL_FEATURES = (metalcenter_descriptors + functionalgroup_descriptors + linker_descriptors + mol_desc)
    df_subset_merged = pd.read_csv('../data/development_set.csv')
    augment_dict = read_pickle('../data/augment_dict.pkl')
    df_train, df_test = train_test_split(df_subset_merged, train_size=0.7)
    df_train = augment_data(df_train, augment_dict)

    X_train = df_train[CHEMICAL_FEATURES]
    y_train = df_train[['r', 'g', 'b']]

    X_test = df_test[CHEMICAL_FEATURES]
    y_test = df_test[['r', 'g', 'b']]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    Y_train = y_train / 255
    Y_test = y_test / 255
    return X_train, Y_train, X_test, Y_test


def keras_fmin_fnct(space):
    X_train, Y_train, X_test, Y_test = data()
    mlp = Sequential()
    dropout = space['dropout']
    l1rate = space['l1rate']
    layers = space['layers']

    mlp.add(
        Dense(
            layers[0],
            activation='linear',
            kernel_initializer='he_normal',
            input_shape=(178,),
            activity_regularizer=l1(l1rate),
        ))
    mlp.add(Activation('relu'))
    mlp.add(Dropout(dropout))

    for layer in layers[1:]:
        mlp.add(Dense(
            layer,
            activation='linear',
            kernel_initializer='he_normal',
            activity_regularizer=l1(l1rate),
        ))
        mlp.add(Activation('relu'))
        mlp.add(Dropout(dropout))

    mlp.add(Dense(3, activation=mapping_to_target_range, kernel_initializer='he_normal'))

    mlp.compile(
        optimizer=Adam(learning_rate=space['learning_rate']),
        loss=space['loss'],
        metrics=['mae', 'mean_absolute_percentage_error'],
    )

    callbacks = []

    if space['loss_1'] == 'es':
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=space['patience'],
            verbose=0,
            mode='auto',
        ))

    # if space["patience_1"] == "lrs":
    #     callbacks.append(
    #         learning_rate_reduction=ReduceLROnPlateau(
    #             monitor="val_loss",
    #             patience=space["patience_2"],
    #             verbose=1,
    #             factor=space["dropout_2"],
    #             min_lr=space["learning_rate_1"],
    #         )
    #     )

    mlp.fit(
        X_train,
        Y_train,
        callbacks=callbacks,
        epoch=100,
        batch_size=space['batch_size'],
        validation_data=(X_test, Y_test),
    )

    _, mae, _ = mlp.evaluate(X_test, Y_test, verbose=0)

    return {'loss': mae, 'status': STATUS_OK, 'model': mlp}


def get_space():
    return {
        'layers':
            hp.choice(
                'layers',
                [
                    [64, 32, 16, 8],
                    [128, 64, 32, 16, 8],
                    [128, 16, 8],
                    [32, 16, 8],
                    [64, 16, 8],
                    [64, 32, 8],
                    [64, 32, 16],
                    [16, 8, 8],
                    [8, 8, 8, 8],
                    [164, 64, 32, 16],
                    [128, 32, 16],
                    [128, 64, 8],
                    [64, 8],
                    [16, 16, 16],
                ],
            ),
        'learning_rate':
            hp.loguniform('learning_rate', -6, -1),
        'loss':
            hp.choice('loss', [huber_fn, 'mae', 'mse']),
        'loss_1':
            hp.choice('loss_1', ['es', False]),
        'patience':
            hp.choice('patience', [10, 20, 30, 40, 60, 80, 100]),
        'l1rate':
            hp.loguniform('l1rate', -6, -1),
        'dropout':
            hp.uniform('dropout', 0, 1),
        'batch_size':
            hp.choice('batch_size', [64, 128, 256]),
    }


def getBestModelfromTrials(trials):
    valid_trial_list = [trial for trial in trials if STATUS_OK == trial['result']['status']]
    losses = [float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    return best_trial_obj['result']['mlp']


if __name__ == '__main__':
    experiment = Experiment(project_name='color-ml')
    with experiment.train():
        trials = Trials()
        best = fmin(keras_fmin_fnct, get_space(), algo=tpe.suggest, max_evals=150, trials=trials)
        X_train, Y_train, X_test, Y_test = data()
        print('Evalutation of best performing model:')

    joblib.dump(best, 'best.joblib')
    model = getBestModelfromTrials(trials)
    joblib.dump(model, 'best_model.joblib')
    experiment.log_asset('best.joblib')
    experiment.log_asset('best_model.joblib')
