# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import subprocess
import time
from pathlib import Path

import click
import ruamel.yaml as yaml

from colorml.utils import (get_timestamp_string, make_if_not_exists, parse_config)

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
srun python -m colorml.run_mlp_training {submission}
"""

scalers = ['minmax', 'standard']
colorspaces = ['rgb', 'hsl']
architectures = [[64, 32, 16, 8], [128, 64, 32, 16, 8], [128, 16, 8], [32, 16, 8]]
lrs = [3e-3, 3e-2, 3e-4]
l1s = [1e-6, 1e-4, 1e-3, 1e-2]
augments = [True]

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
        for j, architecture in enumerate(architectures):
            for k, colorspace in enumerate(colorspaces):
                for l, l1 in enumerate(l1s):
                    for m, lr in enumerate(lrs):
                        for n, feature in enumerate(features):
                            for o, augment in enumerate(augments):

                                basename = '_'.join([
                                    get_timestamp_string(),
                                    str(i),
                                    str(j),
                                    str(k),
                                    str(l),
                                    str(m),
                                    str(n),
                                    str(o),
                                ])
                                configfile = write_config_file(
                                    basename,
                                    scaler,
                                    architecture,
                                    colorspace,
                                    l1,
                                    lr,
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
    architecture,
    colorspace,
    l1,
    lr,
    feature,
    augment,
):
    config = parse_config('/scratch/kjablonk/colorml/colorml/models/models/mlp_base.yaml')
    config['scaler'] = scaler
    config['features'] = feature
    config['model']['units'] = architecture
    config['training']['learning_rate'] = lr
    config['early_stopping']['enabled'] = False
    config['early_stopping']['patience'] = 30
    config['augmentation']['enabled'] = augment
    config['model']['l1'] = l1
    config['model']['kernel_init'] = 'he_normal'
    config['dropout']['probability'] = 0.2
    config['dropout']['gaussian'] = False
    config['colorspace'] = colorspace
    config['tags'] = ['mlp dropout', colorspace, 'augmentation']
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
