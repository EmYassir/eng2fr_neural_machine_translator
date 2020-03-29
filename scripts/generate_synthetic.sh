#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:k80:1
#SBATCH --mem=4G
#SBATCH --time=24:00:00

FOLDER="/project/cq-training-1/project2/teams/team09"
CODE_FOLDER="${FOLDER}/ift6759_project2"

input_file_path=${1}
config_file=${2}
num_lines=${3}

if [ -z "${input_file_path}" ]; then
      echo "Error: \$input_file_path argument is empty"
      exit 1
fi
if [ -z "${config_file}" ]; then
      echo "Error: \$config_file argument is empty"
      exit 1
fi
if [ -z "${num_lines}" ]; then
      echo "Error: \$num_lines argument is empty"
      exit 1
fi

# 1. Create your environement locally
source "${FOLDER}/venv/bin/activate"


cd "${CODE_FOLDER}" || exit
echo "Now in directory ${PWD}"

python -m src.generate_synthetic \
        -i "${input_file_path}" \
        -c "${config_file}" \
        -n "${num_lines}"