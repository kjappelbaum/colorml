# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import concurrent.futures
from functools import partial

import joblib
import numpy as np
import shap
from six.moves import map, range
from tqdm import tqdm

from ..utils.descriptornames import *


def get_one_boostrap_sample(_, explainer, data, samples, feat_filtred_mc_indices, feat_filtred_lc_indices):
    print('entered subroutine')
    indices = np.random.choice(np.arange(len(data) - 1), samples, replace=True)
    shap_val = explainer.shap_values(data[indices])
    norm = np.abs(shap_val).sum(axis=0)

    mc_importance = np.abs(shap_val[:, feat_filtred_mc_indices]).sum() / norm
    lc_importance = np.abs(shap_val[:, feat_filtred_lc_indices]).sum() / norm

    return shap_val, mc_importance, lc_importance


def get_bootstrapped_shap_values(model,
                                 data,
                                 feat_filtred_mc_indices,
                                 feat_filtred_lc_indices,
                                 channel=0,
                                 samples=30,
                                 rounds=5000):
    booster = model.estimators_[channel]
    explainer = shap.TreeExplainer(booster)
    metal_importance = []
    ligand_importance = []
    shap_values = []

    partial_one_boostrap = partial(get_one_boostrap_sample,
                                   data=data,
                                   explainer=explainer,
                                   samples=samples,
                                   feat_filtred_lc_indices=feat_filtred_lc_indices,
                                   feat_filtred_mc_indices=feat_filtred_mc_indices)

    for result in tqdm(list(map(partial_one_boostrap, list(range(rounds)))), total=len(list(range(rounds)))):
        shap_val, mc, lc = result
        metal_importance.append(mc)
        ligand_importance.append(lc)
        shap_values.append(shap_val)

    return shap_values, metal_importance, ligand_importance


def main():
    model = joblib.load('regressor_medianrun_2020_06_08_12_59_1591613951.joblib')
    X_train = np.load('X_train_run_2020_06_08_12_59_1591613951.npy')

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
        'sum-func-chi-0-all',
        'sum-func-chi-1-all',
        'sum-func-chi-2-all',
        'sum-func-chi-3-all',
        'sum-func-Z-0-all',
        'sum-func-Z-1-all',
        'sum-func-Z-2-all',
        'sum-func-Z-3-all',
        'sum-func-I-0-all',
        'sum-func-I-1-all',
        'sum-func-I-2-all',
        'sum-func-I-3-all',
        'sum-func-T-0-all',
        'sum-func-T-1-all',
        'sum-func-T-2-all',
        'sum-func-T-3-all',
        'sum-func-S-0-all',
        'sum-func-S-1-all',
        'sum-func-S-2-all',
        'sum-func-S-3-all',
        'sum-func-alpha-0-all',
        'sum-func-alpha-1-all',
        'sum-func-alpha-2-all',
        'sum-func-alpha-3-all',
        'sum-D_func-chi-1-all',
        'sum-D_func-chi-2-all',
        'sum-D_func-chi-3-all',
        'sum-D_func-Z-1-all',
        'sum-D_func-Z-2-all',
        'sum-D_func-Z-3-all',
        'sum-D_func-T-1-all',
        'sum-D_func-T-2-all',
        'sum-D_func-T-3-all',
        'sum-D_func-S-1-all',
        'sum-D_func-S-2-all',
        'sum-D_func-S-3-all',
        'sum-D_func-alpha-1-all',
        'sum-D_func-alpha-2-all',
        'sum-D_func-alpha-3-all',
        'sum-f-lig-chi-0',
        'sum-f-lig-chi-1',
        'sum-f-lig-chi-2',
        'sum-f-lig-chi-3',
        'sum-f-lig-Z-0',
        'sum-f-lig-Z-1',
        'sum-f-lig-Z-2',
        'sum-f-lig-Z-3',
        'sum-f-lig-I-0',
        'sum-f-lig-I-1',
        'sum-f-lig-I-2',
        'sum-f-lig-I-3',
        'sum-f-lig-T-0',
        'sum-f-lig-T-1',
        'sum-f-lig-T-2',
        'sum-f-lig-T-3',
        'sum-f-lig-S-0',
        'sum-f-lig-S-1',
        'sum-f-lig-S-2',
        'sum-f-lig-S-3',
        'sum-lc-chi-0-all',
        'sum-lc-chi-1-all',
        'sum-lc-chi-2-all',
        'sum-lc-chi-3-all',
        'sum-lc-Z-0-all',
        'sum-lc-Z-1-all',
        'sum-lc-Z-2-all',
        'sum-lc-Z-3-all',
        'sum-lc-I-0-all',
        'sum-lc-I-1-all',
        'sum-lc-I-2-all',
        'sum-lc-I-3-all',
        'sum-lc-T-0-all',
        'sum-lc-T-1-all',
        'sum-lc-T-2-all',
        'sum-lc-T-3-all',
        'sum-lc-S-0-all',
        'sum-lc-S-1-all',
        'sum-lc-S-2-all',
        'sum-lc-S-3-all',
        'sum-lc-alpha-0-all',
        'sum-lc-alpha-1-all',
        'sum-lc-alpha-2-all',
        'sum-lc-alpha-3-all',
        'sum-D_lc-chi-1-all',
        'sum-D_lc-chi-2-all',
        'sum-D_lc-chi-3-all',
        'sum-D_lc-Z-1-all',
        'sum-D_lc-Z-2-all',
        'sum-D_lc-Z-3-all',
        'sum-D_lc-T-1-all',
        'sum-D_lc-T-2-all',
        'sum-D_lc-T-3-all',
        'sum-D_lc-S-1-all',
        'sum-D_lc-S-2-all',
        'sum-D_lc-S-3-all',
        'sum-D_lc-alpha-1-all',
        'sum-D_lc-alpha-2-all',
        'sum-D_lc-alpha-3-all',
        'sum-mc_CRY-chi-0-all',
        'sum-mc_CRY-chi-1-all',
        'sum-mc_CRY-chi-2-all',
        'sum-mc_CRY-chi-3-all',
        'sum-mc_CRY-Z-0-all',
        'sum-mc_CRY-Z-1-all',
        'sum-mc_CRY-Z-2-all',
        'sum-mc_CRY-Z-3-all',
        'sum-mc_CRY-I-0-all',
        'sum-mc_CRY-I-1-all',
        'sum-mc_CRY-I-2-all',
        'sum-mc_CRY-I-3-all',
        'sum-mc_CRY-T-0-all',
        'sum-mc_CRY-T-1-all',
        'sum-mc_CRY-T-2-all',
        'sum-mc_CRY-T-3-all',
        'sum-mc_CRY-S-0-all',
        'sum-mc_CRY-S-1-all',
        'sum-mc_CRY-S-2-all',
        'sum-mc_CRY-S-3-all',
        'sum-D_mc_CRY-chi-1-all',
        'sum-D_mc_CRY-chi-2-all',
        'sum-D_mc_CRY-chi-3-all',
        'sum-D_mc_CRY-Z-1-all',
        'sum-D_mc_CRY-Z-2-all',
        'sum-D_mc_CRY-Z-3-all',
        'sum-D_mc_CRY-T-1-all',
        'sum-D_mc_CRY-T-2-all',
        'sum-D_mc_CRY-T-3-all',
        'sum-D_mc_CRY-S-1-all',
        'sum-D_mc_CRY-S-2-all',
        'sum-D_mc_CRY-S-3-all',
    ]

    feat_filtred_mc = []

    feat_filtred_lc = []

    feat_filtred_mc_indices = []

    feat_filtred_lc_indices = []

    for i, feat in enumerate(CHEMICAL_FEATURES):
        if feat in metalcenter_descriptors:
            feat_filtred_mc.append(feat)
            feat_filtred_mc_indices.append(i)
        elif feat in linker_descriptors or feat:
            feat_filtred_lc.append(feat)
            feat_filtred_lc_indices.append(i)
        elif feat in mol_desc:
            feat_filtred_lc.append(feat)
            feat_filtred_lc_indices.append(i)
        elif feat in functionalgroup_descriptors:
            feat_filtred_lc.append(feat)
            feat_filtred_lc_indices.append(i)

    shap_red, mc_red, lc_red = get_bootstrapped_shap_values(model,
                                                            X_train,
                                                            channel=0,
                                                            feat_filtred_lc_indices=feat_filtred_lc_indices,
                                                            feat_filtred_mc_indices=feat_filtred_mc_indices)

    np.save('shap_red.npy', shap_red)
    np.save('mc_red.npy', mc_red)
    np.save('lc_red.npy', lc_red)


if __name__ == '__main__':
    main()
