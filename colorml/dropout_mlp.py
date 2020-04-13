# -*- coding: utf-8 -*-
"""Build MLP and use, following Gal and Ghahramani (2016) https://arxiv.org/pdf/1506.02142.pdf, 
the dropout as an approximation for Bayesian inference. 
I'd recommend sticking to ReLU activation, as this is what they originally explored (see also Gal's blog posts)
This method is still debated, but it is much easier to train than variational models. 
"""
from comet_ml import Experiment
from keras import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.constraints import MinMaxNorm
from keras.initializers import Constant
from keras.layers import Dropout, Dense, BatchNormalization, GaussianDropout, Activation
from keras.optimizers import Adam
import tensorflow as tf
from numpy.random import seed


from colorml.utils import (
    mapping_to_target_range,
    get_timestamp_string,
    plot_predictions,
    huber_fn,
    mapping_to_target_range_sig,
    read_pickle,
    augment_data,
)


def build_model(
    n_features,
    layers=[64, 32, 16, 8],
    dropout: float = 0.2,
    gaussian_dropout: bool = False,
    kernel_init="he_normal",
    l1rate: float = 0.001,
):
    mlp = Sequential()
    # http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf
    mlp.add(
        Dense(
            layers[0],
            activation="linear",
            kernel_initializer=kernel_init,
            input_shape=(n_features,),
            activity_regularizer=l1(l1rate),
        )
    )
    mlp.add(Activation("relu"))
    if not gaussian_dropout:
        mlp.add(Dropout(dropout))
    else:
        mlp.add(GaussianDropout(dropout))

    for layer in layers[1:]:
        mlp.add(
            Dense(
                layer,
                activation="linear",
                kernel_initializer=kernel_init,
                activity_regularizer=l1(l1rate),
            )
        )
        mlp.add(Activation("relu"))
        if not gaussian_dropout:
            mlp.add(Dropout(dropout))
        else:
            mlp.add(GaussianDropout(dropout))

    mlp.add(
        Dense(3, activation=mapping_to_target_range, kernel_initializer=kernel_init)
    )

    return mlp


def train_model(
    experiment,
    mlp,
    train_data: tuple,
    valid_data: tuple,
    logger,
    loss=huber_fn,
    lr: float = 3e-3,
    epochs: int = 500,
    batch_size: int = 264,
    early_stopping: int = None,
    reduce_lr: dict = None,
    random_seed: int = 82199,
):
    seed(random_seed)
    tf.random.set_seed(random_seed)

    X_train, y_train = train_data
    X_valid, y_valid = valid_data

    assert len(X_train) == len(y_train)
    assert len(X_valid) == len(y_valid)

    # We normalize y for now
    assert y_train.max() <= 1
    assert y_train.min() >= 0
    assert y_valid.max() <= 1
    assert y_valid.min() >= 0

    assert X_train.shape[1] == X_valid.shape[1]

    logger.info("Will now start training.")

    mlp.compile(
        optimizer=Adam(learning_rate=lr),
        loss=huber_fn,
        metrics=["mae", "mean_absolute_percentage_error"],
    )

    callbacks = []

    if isinstance(early_stopping, int):
        callbacks.append(
            EarlyStopping(
                monitor="val_loss", patience=early_stopping, verbose=0, mode="auto"
            )
        )

    if isinstance(reduce_lr, dict):
        callbacks.append(
            learning_rate_reduction=ReduceLROnPlateau(
                monitor="val_loss",
                patience=reduce_lr["patience"],
                verbose=1,
                factor=reduce_lr["factor"],
                min_lr=reduce_lr["min_lr"],
            )
        )

    with experiment.train():
        history = mlp.fit(
            X_train,
            y_train,
            callbacks=callbacks,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_valid, y_valid),
        )

    return mlp
