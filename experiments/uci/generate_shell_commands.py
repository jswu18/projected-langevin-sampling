import argparse
import os

from experiments.uci.schemas import DatasetSchema
from experiments.utils import create_directory

parser = argparse.ArgumentParser(
    description="Generate shell commands for running experiments on UCI datasets. This includes commands for the "
    "Myriad cluster. "
)
parser.add_argument(
    "--number_of_seeds",
    type=int,
    required=False,
    default=5,
    help="Number of seeds to run for each experiment.",
)
parser.add_argument(
    "--max_runtime",
    type=str,
    required=False,
    default="48:00:00",
    help="Max runtime.",
)
parser.add_argument(
    "--num_gpus",
    type=int,
    required=False,
    default=0,
    help="Number of GPUs.",
)
parser.add_argument(
    "--num_cores",
    type=int,
    required=False,
    default=16,
    help="Number of cores.",
)
parser.add_argument(
    "--mem",
    type=str,
    required=False,
    default="5G",
    help="Amount of RAM per core.",
)
parser.add_argument(
    "--username",
    type=str,
    required=False,
    default="ucabwuh",
    help="Myraid username.",
)


def construct_shell_command(repository_path: str, dataset_name: str, seed: int) -> str:
    return "\n".join(
        [
            f"cd {repository_path}",
            "export PYTHONPATH=$PWD",
            f"python experiments/uci/main.py --dataset_name {dataset_name} --data_seed {seed} --config_path "
            f"experiments/uci/config.yaml",
        ]
    )


def _build_base_myriad_commands(
    job_name: str,
    max_runtime: str,
    num_gpus: int,
    num_cores: int,
    mem: str,
    user_name: str,
    repository_path: str,
):
    base_myriad_commands = [
        "#$ -S /bin/bash",
        "# Max runtime of job. Shorter means scheduled faster.",
        f"#$ -l h_rt={max_runtime}",
        f"#$ -l gpu={num_gpus}",
        "# Number of cores. Set to more than one when using a GPU, but not too many.",
        f"#$ -pe smp {num_cores}",
        "# Amount of RAM.",
        "# IMPORTANT: each core gets RAM below, so in this case 20GB total.",
        f"#$ -l mem={mem}",
        "# Stops smaller jobs jumping in front of you in the queue.",
        "#$ -R y",
        "# Merges the standard output and error output files into one file.",
        "#$ -j y",
        "# Working directory of job.",
        "# (-cwd is short for set wd to the directory you run qsub)",
        f"#$ -wd /home/{user_name}/function-space-gradient-flow",
        "# Name of the job",
        f"#$ -N {job_name}",
        "date",
        "nvidia-smi",
        "module load python3/3.11",
        f"cd {repository_path}",
        "source .venv/bin/activate",
    ]
    return base_myriad_commands


if __name__ == "__main__":
    args = parser.parse_args()
    shell_command_dir = "experiments/uci/shell_commands"
    create_directory(shell_command_dir)
    repository_path_ = os.getcwd()
    shell_commands = []
    myriad_command_paths = []
    for dataset_name_ in DatasetSchema.__members__:
        for seed_ in range(args.number_of_seeds):
            shell_command_ = construct_shell_command(
                repository_path=repository_path_,
                dataset_name=dataset_name_,
                seed=seed_,
            )
            shell_command_path = os.path.join(
                shell_command_dir, f"{dataset_name_}-{seed_}.sh"
            )
            with open(shell_command_path, "w") as file:
                file.write(shell_command_)
            shell_commands.append(shell_command_)

            base_myriad_commands_ = _build_base_myriad_commands(
                job_name=f"{dataset_name_}-{seed_}",
                max_runtime=args.max_runtime,
                num_gpus=args.num_gpus,
                num_cores=args.num_cores,
                mem=args.mem,
                user_name=args.username,
                repository_path=repository_path_,
            )

            myriad_command_path = os.path.join(
                shell_command_dir, f"{dataset_name_}-{seed_}.myriad.sh"
            )
            with open(myriad_command_path, "w") as file:
                file.write("\n".join(base_myriad_commands_ + [shell_command_]))
            myriad_command_paths.append(myriad_command_path)

    with open("uci.sh", "w") as file:
        file.write("\n".join([shell_command for shell_command in shell_commands]))

    with open("myriad_qsubs_uci.sh", "w") as file:
        file.write(
            "\n".join(
                [
                    "qsub " + myriad_command_path
                    for myriad_command_path in myriad_command_paths
                ]
            )
        )
