import os
import time
import click
import subprocess
from pathlib import Path
import ruamel.yaml as yaml
from colorml.utils import parse_config, get_timestamp_string, make_if_not_exists

BASEFOLDER = "/scratch/kjablonk/colorml/colorml"
SUBMISSION = """#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=5:00:0
#SBATCH --qos=gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu

module load cuda cudnn 
source ~/anaconda3/bin/activate colorml
srun python -m colorml.run_training {submission}
"""

scalers = ["minmax", "standard"]
activations = ["relu", "selu", "elu"]
architectures = [
    ([16, 8], [8, 8, 3]),
    ([32, 16, 8], [8, 8, 3]),
    ([32, 8, 8], [8, 8, 4, 3]),
    ([64, 8, 8], [8, 8, 3]),
    ([64, 8], [8, 8, 8, 3]),
    ([64, 16], [16, 8, 8, 3]),
    ([64, 16], [16, 8, 3]),
]


@click.command("cli")
@click.option("--submit", is_flag=True)
def main(submit=False):
    for i, scaler in enumerate(scalers):
        for j, activation in enumerate(activations):
            for k, architecture in enumerate(architectures):
                basename = "_".join([get_timestamp_string(), str(i), str(j), str(k)])
                configfile = write_config_file(
                    basename, scaler, activation, architecture
                )
                slurmfile = write_submission_script(configfile, basename)

                if submit:
                    subprocess.call(
                        "sbatch {}".format("{}".format(slurmfile)),
                        shell=True,
                        cwd=BASEFOLDER,
                    )
                    time.sleep(2)


def write_submission_script(configfile, basename):
    submission = SUBMISSION.format(submission=str(configfile))
    slurmfile = os.path.join(BASEFOLDER, basename + ".slurm")
    with open(slurmfile, "w") as fh:
        fh.write(submission)
    return slurmfile


def write_config_file(basename, scaler, activation, architecture):
    config = parse_config(
        "/scratch/kjablonk/colorml/colorml/models/models/test_config.yaml"
    )
    config["scaler"] = scaler
    config["model"]["activation_function"] = activation
    config["model"]["units"] = architecture[0]
    config["model"]["head_units"] = architecture[1]
    config["training"]["cycling_lr"] = False
    config["training"]["kl_annealing"] = True
    config["early_stopping"]["patience"] = 25
    config["augmentation"]["enabled"] = False
    outpath = os.path.join(BASEFOLDER, "results", "models", basename)
    make_if_not_exists(outpath)
    config["outpath"] = outpath
    outname = os.path.join(BASEFOLDER, "models/models/", basename + ".yaml")
    with open(outname, "w",) as outfile:
        yaml.dump(config, outfile)

    return outname


if __name__ == "__main__":
    main()
