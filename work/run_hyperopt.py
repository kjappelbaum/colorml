# -*- coding: utf-8 -*-
import pandas as pd
from comet_ml import Experiment
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import Trials, STATUS_OK, tpe
import joblib

from keras.models import Model
from keras.layers import Input, Dense, Dropout, concatenate, Activation
from keras.optimizers import RMSprop, Adam

from keras.datasets import mnist
from keras.utils import np_utils
from keras.regularizers import l1
import tensorflow as tf
from keras.layers import (
    Dropout,
    Dense,
    BatchNormalization,
    GaussianDropout,
    GaussianNoise,
    LeakyReLU,
)

lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)
from keras import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# from livelossplot.keras import PlotLossesCallback
from keras.constraints import MinMaxNorm
from keras.initializers import Constant
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sys

sys.path.append("../")
from colorml.utils import (
    mapping_to_target_range,
    read_pickle,
    get_timestamp_string,
    plot_predictions,
    huber_fn,
    mapping_to_target_range_sig,
    augment_data,
)
from colorml.descriptornames import *


def data():
    CHEMICAL_FEATURES = (
        metalcenter_descriptors
        + functionalgroup_descriptors
        + linker_descriptors
        + mol_desc
    )
    df_subset_merged = pd.read_csv("../data/color_feat_merged.csv")
    augment_dict = read_pickle("../data/augment_dict.pkl")
    df_train, df_test = train_test_split(df_subset_merged, train_size=0.7)
    df_train = augment_data(df_train, augment_dict)

    X_train = df_train[CHEMICAL_FEATURES]
    y_train = df_train[["r", "g", "b"]]

    X_test = df_test[CHEMICAL_FEATURES]
    y_test = df_test[["r", "g", "b"]]

    name_train = df_train["color_cleaned"]
    name_test = df_test["color_cleaned"]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_test, X_valid, y_test, y_valid, names_valid, names_valid = train_test_split(
        X_test, y_test, name_test, train_size=0.9
    )

    Y_train = y_train / 255
    Y_valid = y_valid / 255
    Y_test = y_test / 255
    return X_train, Y_train, X_test, Y_test


def keras_fmin_fnct(space):
    X_train, Y_train, X_test, Y_test = data()
    mlp = Sequential()
    dropout = space["dropout"]
    l1rate = space["dropout_1"]
    layers = space["layers"]

    mlp.add(
        Dense(
            layers[0],
            activation="linear",
            kernel_initializer="he_normal",
            input_shape=(178,),
            activity_regularizer=l1(l1rate),
        )
    )
    mlp.add(Activation("relu"))
    mlp.add(Dropout(dropout))

    for layer in layers[1:]:
        mlp.add(
            Dense(
                layer,
                activation="linear",
                kernel_initializer="he_normal",
                activity_regularizer=l1(l1rate),
            )
        )
        mlp.add(Activation("relu"))
        mlp.add(Dropout(dropout))

    mlp.add(
        Dense(3, activation=mapping_to_target_range, kernel_initializer="he_normal")
    )

    mlp.compile(
        optimizer=Adam(learning_rate=space["learning_rate"]),
        loss=space["loss"],
        metrics=["mae", "mean_absolute_percentage_error"],
    )

    callbacks = []

    if space["loss_1"] == "es":
        callbacks.append(
            EarlyStopping(
                monitor="val_loss", patience=space["patience"], verbose=0, mode="auto",
            )
        )

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
        epochs=500,
        batch_size=space["batch_size"],
        validation_data=(X_test, Y_test),
    )

    score, mae, mape = mlp.evaluate(X_test, Y_test, verbose=0)

    return {"loss": mae, "status": STATUS_OK, "model": mlp}


def get_space():
    return {
        "dropout": hp.uniform("dropout", 0, 1),
        "dropout_1": hp.uniform("dropout_1", 0, 1),
        "layers": hp.choice(
            "layers",
            [
                [64, 32, 16, 8],
                [128, 64, 32, 16, 8],
                [128, 16, 8],
                [32, 16, 8],
                [16, 8, 8],
                [8, 8, 8, 8],
                [164, 64, 32, 16],
                [128, 32, 16],
                [128, 64, 8],
                [64, 8],
                [16, 16, 16],
            ],
        ),
        "learning_rate": hp.uniform("learning_rate", 1e-5, 1e-2),
        "loss": hp.choice("loss", [huber_fn, "mae", "mse"]),
        "loss_1": hp.choice("loss_1", ["es", False]),
        "patience": hp.choice("patience", [10, 20, 30, 40, 60, 80, 100]),
        # "patience_1": hp.choice("patience_1", ["lrs", False]),
        "patience_2": hp.choice("patience_2", [10, 20, 30, 40, 60, 80, 100]),
        "dropout_2": hp.uniform("dropout_2", 0, 1),
        "learning_rate_1": hp.uniform("learning_rate_1", 1e-5, 1e-2),
        "batch_size": hp.choice("batch_size", [64, 128, 256]),
    }


if __name__ == "__main__":
    experiment = Experiment(project_name="color-ml")
    with experiment.train():
        trials = Trials()
        best = fmin(
            keras_fmin_fnct, get_space(), algo=tpe.suggest, max_evals=50, trials=trials
        )
        X_train, Y_train, X_test, Y_test = data()
        print("Evalutation of best performing model:")

    joblib.dump(best, "best.joblib")
    experiment.log_asset("best.joblib")
