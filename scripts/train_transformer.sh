#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:k80:1
#SBATCH --mem=4G
#SBATCH --time=1:00:00

FOLDER="/project/cq-training-1/project2/teams/team09"
CODE_FOLDER="${FOLDER}/ift6759_project2"
ZIP_FILE="data.zip"
config_file=${1}
cfg_path="${CODE_FOLDER}/config_files/${config_file}"

# Check if config file is valid
if [ -z "${config_file}" ]; then
      echo "Error: \$config_file argument is empty"
      exit 1
fi
if [ ! -e "${cfg_path}" ]; then
    echo "Error: cfg_path=${cfg_path} does not exist"
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

# 4. Launch your job, and look for the dataset into $SLURM_TMPDIR
python -m src.train_transformer \
        --cfg_path "${cfg_path}" \
        --data_path "${SLURM_TMPDIR}" \
        --save_path "${CODE_FOLDER}"