# -*- coding: utf-8 -*-
import click
import comet_ml
import os
import dill
import pickle
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
from colour.models import (
    RGB_to_XYZ,
    XYZ_to_xy,
    XYZ_to_Lab,
    XYZ_to_RGB,
    Lab_to_XYZ,
    RGB_to_HSL,
    RGB_to_HSV,
    HSL_to_RGB,
)
from colour.plotting import filter_RGB_colourspaces
from colour.utilities import first_item

colourspace = first_item(filter_RGB_colourspaces("sRGB").values())
import pandas as pd
from numpy.random import seed
import joblib
from .descriptornames import *
import logging


def orchestrate(config, configfile):
    logger = logging.getLogger("bayesnet")
    logger.setLevel(logging.DEBUG)
    experiment = Experiment(project_name="color-ml")
    experiment.log_asset(configfile)
    experiment.log_parameters(flatten(config))
    experiment.log_asset(config["data"])
    experiment.log_parameter(
        name="total_depth",
        value=len(config["model"]["units"]) + len(config["model"]["head_units"]),
    )
    experiment.log_parameter(name="widest_layer", value=config["model"]["units"][0])
    # using now for loop because the add.tags() function did not work
    for tag in config["tags"]:
        experiment.add_tag(tag)
    seed(int(config["seed"]))

    make_if_not_exists(config["outpath"])

    if config["early_stopping"]["enabled"] is True:
        patience = int(config["early_stopping"]["patience"])
    else:
        patience = None

    df = pd.read_csv(config["data"])

    df_train, df_test = train_test_split(
        df, train_size=config["train_size"], random_state=int(config["seed"])
    )

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

    if config["colorspace"] == "rgb":
        pass
    elif config["colorspace"] == "hsl":
        y_train = RGB_to_HSL(y_train)
        y_test = RGB_to_HSL(y_test)
    elif config["colorspace"] == "hsv":
        y_train = RGB_to_HSV(y_train)
        y_test = RGB_to_HSV(y_train)
    elif config["colorspace"] == "lab":
        train_xyz = RGB_to_XYZ(
            y_train,
            colourspace.whitepoint,
            colourspace.whitepoint,
            colourspace.RGB_to_XYZ_matrix,
        )
        test_xyz = RGB_to_XYZ(
            y_test,
            colourspace.whitepoint,
            colourspace.whitepoint,
            colourspace.RGB_to_XYZ_matrix,
        )

        y_train = XYZ_to_Lab(train_xyz)
        y_test = XYZ_to_Lab(test_xyz)

        y_train = (y_train + [0, 100, 100]) / [100, 200, 200]
        y_test = (y_test + [0, 100, 100]) / [100, 200, 200]

    joblib.dump(scaler, os.path.join(config["outpath"], "scaler.joblib"))
    experiment.log_asset(os.path.join(config["outpath"], "scaler.joblib"))

    X_test, X_valid, y_test, y_valid = train_test_split(
        X_test,
        y_test,
        train_size=config["valid_size"],
        random_state=int(config["seed"]),
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

    if config["training"]["cycling_lr"]:
        cycling_lr = True
    else:
        cycling_lr = False

    if config["training"]["kl_annealing"]:
        if config["kl_anneal"]["method"] == "linear":
            kl_annealing = {
                "method": "linear",
                "M": int(config["kl_anneal"]["constant"]),
            }
        elif config["kl_anneal"]["method"] == "tanh":
            kl_annealing = {
                "method": "tanh",
                "M": int(config["kl_anneal"]["constant"]),
            }
        elif config["kl_anneal"]["method"] == "cycling":
            kl_annealing = {
                "method": "cycling",
                "M": int(config["kl_anneal"]["constant"]),
            }

    else:
        kl_annealing = None

    logger.info("Built model.")
    logger.info(f"Head units: {config['model']['head_units']}")
    logger.info(f"Head units: {config['model']['units']}")

    # experiment.log_asset_data(
    #     (X_train, y_train), metadata={"split": "train", "scaled": True}
    # )
    # experiment.log_asset_data(
    #     (X_valid, y_valid), metadata={"split": "valid", "scaled": True}
    # )
    # experiment.log_asset_data(
    #     (X_test, y_test), metadata={"split": "test", "scaled": True}
    # )

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
        cycling_lr=cycling_lr,
        kl_annealing=kl_annealing,
    )

    with open(os.path.join(config["outpath"], "model.dill"), "wb") as fh:
        dill.dump(model, fh)

    experiment.log_asset(os.path.join(config["outpath"], "model.dill"))

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
    orchestrate(config, path)


if __name__ == "__main__":
    main()
