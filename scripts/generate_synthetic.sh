#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:k80:1
#SBATCH --mem=4G
#SBATCH --time=24:00:00

FOLDER="/project/cq-training-1/project2/teams/team09"
CODE_FOLDER="${FOLDER}/ift6759_project2"
ZIP_FILE="data.zip"

name_input_file=${1}
config_file=${2}
num_lines=${3}

if [ -z "${name_input_file}" ]; then
      echo "Error: \$name_input_file argument is empty"
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

# 2. Copy your dataset on the compute node
cp "${FOLDER}/${ZIP_FILE}" "${SLURM_TMPDIR}"

# 3. Eventually unzip your dataset
unzip "${SLURM_TMPDIR}/${ZIP_FILE}" -d "${SLURM_TMPDIR}"

cd "${CODE_FOLDER}" || exit
echo "Now in directory ${PWD}"

python -m src.generate_synthetic \
        -i "${SLURM_TMPDIR}/data/${name_input_file}" \
        -c "${config_file}" \
        -n "${num_lines}"