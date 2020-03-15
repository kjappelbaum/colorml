# -*- coding: utf-8 -*-
import click
import comet_ml
import os
from comet_ml import Experiment
from .utils import (
    parse_config,
    make_if_not_exists,
    read_pickle,
    select_features,
    augment_data,
    flatten,
)
from .bayesiannet import build_model, train_model, measure_performance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy.random import seed
import joblib
from .descriptornames import *
import logging


def orchestrate(config):
    logger = logging.getLogger("bayesnet")
    logger.setLevel(logging.DEBUG)
    experiment = Experiment(project_name="color-ml")
    experiment.log_parameters(flatten(config))
    experiment.log_asset(config["data"])

    seed(int(config["seed"]))

    make_if_not_exists(config["outpath"])

    if config["early_stopping"]["enabled"] is True:
        patience = int(config["early_stopping"]["patience"])
    else:
        patience = None

    df = pd.read_csv(config["data"])

    df_train, df_test = train_test_split(df, train_size=config["train_size"])

    if config["augmentation"]["enabled"]:
        augment_dict = read_pickle(config["augmentation"]["augmentation_dict"])
        experiment.log_asset(config["augmentation"]["augmentation_dict"])
        df_train = augment_data(df_train, augment_dict)

    features = select_features(config["features"])
    X_train = df_train[features].values
    y_train = df_train[["r", "g", "b"]] / 255

    X_test = df_test[features].values
    y_test = df_test[["r", "g", "b"]] / 255

    name_train = df_train["color_cleaned"]
    name_test = df_test["color_cleaned"]

    if config["scaler"] == "minmax":
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif config["scaler"] == "standard":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    y_train = y_train.values
    y_test = y_test.values

    joblib.dump(scaler, os.path.join(config["outpath"], "scaler.joblib"))
    experiment.log_asset(os.path.join(config["outpath"], "scaler.joblib"))

    X_test, X_valid, y_test, y_valid = train_test_split(
        X_test, y_test, train_size=config["valid_size"]
    )

    config["model"]["units"].insert(0, X_train.shape[1])

    model = build_model(
        config["model"]["units"],
        config["model"]["head_units"],
        config["model"]["activation_function"],
        config["model"]["batchnorm"],
    )

    if config["training"]["learning_rate"] != "None":
        lr = float(config["training"]["learning_rate"])
    else:
        lr = None

    logger.info("Built model.")
    logger.info(f"Head units: {config['model']['head_units']}")
    logger.info(f"Head units: {config['model']['units']}")

    model = train_model(
        experiment,
        model,
        (X_train, y_train),
        (X_valid, y_valid),
        logger=logger,
        early_stopping_patience=patience,
        random_seed=int(config["seed"]),
        lr=lr,
        epochs=int(config["training"]["epochs"]),
        batch_size=int(config["training"]["batch_size"]),
    )

    joblib.dump(scaler, os.path.join(config["outpath"], "model.joblib"))
    experiment.log_asset(os.path.join(config["outpath"], "model.joblib"))

    train_performance = measure_performance(model, X_train, y_train)
    experiment.log_metrics(train_performance, prefix="train")
    test_performance = measure_performance(model, X_test, y_test)
    experiment.log_metrics(test_performance, prefix="test")
    results = {"train": train_performance, "test": test_performance}

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(config["outpath"], "results.csv"))


@click.command("cli")
@click.argument("path")
def main(path):
    config = parse_config(path)
    orchestrate(config)


if __name__ == "__main__":
    main()
