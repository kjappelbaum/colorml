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

slmodules -s x86_E5v2_Mellanox_GPU -v
module load gcc cuda cudnn 
source ~/anaconda3/bin/activate colorml
srun python -m colorml.run_training {submission}
"""

scalers = ["minmax"]
activations = ["selu"]
colorspaces = ["hsl", "rgb", "lab"]
kl_anneal_const = [20, 100, 200]
kl_anneal_method = ["linear", "tanh"]
architectures = [
    ([16, 8], [8, 8, 3]),
    ([64, 16], [16, 8, 3]),
]
lrs = [1e-4, 3e-4, 3e-3, 3e-2]


@click.command("cli")
@click.option("--submit", is_flag=True)
def main(submit=False):
    for i, scaler in enumerate(scalers):
        for j, activation in enumerate(activations):
            for k, architecture in enumerate(architectures):
                for l, colorspace in enumerate(colorspaces):
                    for m, annealconst in enumerate(kl_anneal_const):
                        for n, lr in enumerate(lrs):
                            for o, kl_method in enumerate(kl_anneal_method):
                                basename = "_".join(
                                    [
                                        get_timestamp_string(),
                                        str(i),
                                        str(j),
                                        str(k),
                                        str(l),
                                        str(m),
                                        str(n),
                                    ]
                                )
                                configfile = write_config_file(
                                    basename,
                                    scaler,
                                    activation,
                                    architecture,
                                    colorspace,
                                    annealconst,
                                    lr,
                                    kl_method,
                                )
                                slurmfile = write_submission_script(
                                    configfile, basename
                                )

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


def write_config_file(
    basename, scaler, activation, architecture, colorspace, klanneal, lr, kl_method
):
    config = parse_config(
        "/scratch/kjablonk/colorml/colorml/models/models/test_config.yaml"
    )
    config["scaler"] = scaler
    config["model"]["activation_function"] = activation
    config["model"]["units"] = architecture[0]
    config["model"]["head_units"] = architecture[1]
    config["training"]["cycling_lr"] = False
    config["training"]["kl_annealing"] = True
    config["training"]["learning_rate"] = lr
    config["early_stopping"]["patience"] = 30
    config["augmentation"]["enabled"] = False
    config["kl_anneal"] = {"method": kl_method, "constant": klanneal}
    config["colorspace"] = colorspace
    config["tags"] = ["tanh kl anneal", "cycling lr", colorspace, "early stopping"]
    outpath = os.path.join(BASEFOLDER, "results", "models", basename)
    make_if_not_exists(outpath)
    config["outpath"] = outpath
    outname = os.path.join(BASEFOLDER, "models/models/", basename + ".yaml")
    with open(outname, "w",) as outfile:
        yaml.dump(config, outfile)

    return outname


if __name__ == "__main__":
    main()
