# -*- coding: utf-8 -*-
"""Submit the BNN training on the Deneb HPC cluster at EPFL using the GPU partition"""
from __future__ import absolute_import

import os
import subprocess
import time
from pathlib import Path

import click
import ruamel.yaml as yaml

from ..utils.utils import (get_timestamp_string, make_if_not_exists, parse_config)

BASEFOLDER = '/scratch/kjablonk/colorml/colorml'
SUBMISSION = """#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=5:00:0
#SBATCH --qos=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu

slmodules -s x86_E5v2_Mellanox_GPU -v
module load gcc cuda cudnn
source ~/anaconda3/bin/activate colorml
srun python -m colorml.run_training {submission}
"""

scalers = ['minmax', 'standard']
activations = ['selu']
colorspaces = ['rgb']
kl_anneal_const = [100]
kl_anneal_method = ['tanh']
architectures = [
    ([16, 8], [8, 8, 3]),
]
lrs = [3e-3]

features = [
    [
        'metalcenter_descriptors',
        'functionalgroup_descriptors',
        'linker_descriptors',
        'mol_desc',
        'summed_linker_descriptors',
        'summed_metalcenter_descriptors',
        'summed_functionalgroup_descriptors',
    ],
    [
        'metalcenter_descriptors',
        'functionalgroup_descriptors',
        'linker_descriptors',
        'mol_desc',
    ],
    [
        'metalcenter_descriptors',
        'functionalgroup_descriptors',
        'linker_descriptors',
        'summed_linker_descriptors',
        'summed_metalcenter_descriptors',
        'summed_functionalgroup_descriptors',
    ],
    ['metalcenter_descriptors', 'functionalgroup_descriptors', 'linker_descriptors'],
]


@click.command('cli')
@click.option('--submit', is_flag=True)
def main(submit=False):
    for i, scaler in enumerate(scalers):
        for j, activation in enumerate(activations):
            for k, architecture in enumerate(architectures):
                for l, colorspace in enumerate(colorspaces):
                    for m, annealconst in enumerate(kl_anneal_const):
                        for n, lr in enumerate(lrs):
                            for o, kl_method in enumerate(kl_anneal_method):
                                for p, feature in enumerate(features):
                                    for q, augment in enumerate([True, False]):
                                        basename = '_'.join([
                                            get_timestamp_string(),
                                            str(i),
                                            str(j),
                                            str(k),
                                            str(l),
                                            str(m),
                                            str(n),
                                            str(o),
                                            str(p),
                                            str(q),
                                        ])
                                        configfile = write_config_file(
                                            basename,
                                            scaler,
                                            activation,
                                            architecture,
                                            colorspace,
                                            annealconst,
                                            lr,
                                            kl_method,
                                            feature,
                                            augment,
                                        )
                                        slurmfile = write_submission_script(configfile, basename)

                                        if submit:
                                            subprocess.call(
                                                'sbatch {}'.format('{}'.format(slurmfile)),
                                                shell=True,
                                                cwd=BASEFOLDER,
                                            )
                                            time.sleep(2)


def write_submission_script(configfile, basename):
    submission = SUBMISSION.format(submission=str(configfile))
    slurmfile = os.path.join(BASEFOLDER, basename + '.slurm')
    with open(slurmfile, 'w') as fh:
        fh.write(submission)
    return slurmfile


def write_config_file(
    basename,
    scaler,
    activation,
    architecture,
    colorspace,
    klanneal,
    lr,
    kl_method,
    feature,
    augment,
):
    config = parse_config('/scratch/kjablonk/colorml/colorml/models/models/test_config.yaml')
    config['scaler'] = scaler
    config['features'] = feature
    config['model']['activation_function'] = activation
    config['model']['units'] = architecture[0]
    config['model']['head_units'] = architecture[1]
    config['training']['cycling_lr'] = False
    config['training']['kl_annealing'] = True
    config['training']['learning_rate'] = lr
    config['early_stopping']['enabled'] = False
    config['augmentation']['enabled'] = augment
    config['kl_anneal'] = {'method': kl_method, 'constant': klanneal}
    config['colorspace'] = colorspace
    config['tags'] = ['tanh kl anneal', colorspace, 'augmentation']
    outpath = os.path.join(BASEFOLDER, 'results', 'models', basename)
    make_if_not_exists(outpath)
    config['outpath'] = outpath
    outname = os.path.join(BASEFOLDER, 'models/models/', basename + '.yaml')
    with open(
            outname,
            'w',
    ) as outfile:
        yaml.dump(config, outfile)

    return outname


if __name__ == '__main__':
    main()
