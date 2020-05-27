# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import sys

import click
import joblib
import numpy as np
import pandas as pd
from comet_ml import Experiment
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import (DotProduct, ExpSineSquared, Matern, RationalQuadratic, WhiteKernel)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from colorml.descriptornames import *
from colorml.utils import (augment_data, get_timestamp_string, huber_fn, mapping_to_target_range,
                           mapping_to_target_range_sig, plot_predictions, read_pickle)
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

sys.path.append('../')

CHEMICAL_FEATURES = (metalcenter_descriptors + functionalgroup_descriptors + linker_descriptors + mol_desc +
                     summed_functionalgroup_descriptors + summed_linker_descriptors + summed_metalcenter_descriptors)


def get_data(
    df,
    outdir,
    augment=False,
    augment_dict='/scratch/kjablonk/colorml/colorml/data/augment_dict.pkl',
):
    if augment:
        augment_dict = read_pickle(augment_dict)
        df = augment_data(df, augment_dict)

    X_train = df[CHEMICAL_FEATURES]
    y_train = df[['r', 'g', 'b']]
    vt = VarianceThreshold(0.7)
    X_train = vt.fit_transform(X_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    y_train = y_train / 255 - 0.5

    joblib.dump(vt, os.path.join(outdir, 'vt.joblib'))
    joblib.dump(scaler, os.path.join(outdir, 'scaler.joblib'))

    return X_train, y_train


def train(X, y, outdir, max_feat=30):
    experiment = Experiment(project_name='color-ml')

    with experiment.train():
        gp_kernel = RationalQuadratic(length_scale=0.1, length_scale_bounds=(1e-4, 0.5)) + WhiteKernel(
            0.01, (1e-3, 0.5e-1))
        gp = GaussianProcessRegressor(kernel=gp_kernel, n_restarts_optimizer=15, normalize_y=True)

        sfs = SFS(
            gp,
            k_features=max_feat,
            forward=True,
            floating=False,
            scoring='neg_mean_squared_error',
            cv=5,
            verbose=2,
            n_jobs=-1,
        )

        sfs = sfs.fit(X, y)

        joblib.dump(sfs, os.path.join(outdir, 'sfs.joblib'))

    return sfs


@click.command('cli')
@click.argument('infile')
@click.argument('outdir')
def main(infile, outdir):
    df = pd.read_csv(infile)
    X_train, y_train = get_data(df, outdir)
    train(X_train, y_train, outdir)


if __name__ == '__main__':
    main()
