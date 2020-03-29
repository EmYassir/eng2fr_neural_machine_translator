#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:k80:1
#SBATCH --mem=8G
#SBATCH --time=1:00:00

FOLDER="/project/cq-training-1/project2/teams/team09"
CODE_FOLDER="${FOLDER}/ift6759_project2"

input_file_path=${1}
target_file_path=${2}
print_all_scores=${3}
config_file="transformer_eval_cfg.json"

if [ -z "${input_file_path}" ]; then
      echo "Error: \$input_file_path argument is empty"
      exit 1
fi
if [ -z "${target_file_path}" ]; then
      echo "Error: \$target_file_path argument is empty"
      exit 1
fi

# 1. Create your environement locally
source "${FOLDER}/venv/bin/activate"


cd "${CODE_FOLDER}" || exit
echo "Now in directory ${PWD}"

if [ -z "${print_all_scores}" ]; then
      echo "Warning: \$print_all_scores argument is empty -> Defaults to False"
      python -m src.evaluator \
        --input-file-path "${input_file_path}" \
        --target-file-path "${target_file_path}" \
        --config_file "${config_file}"
else
  python -m src.evaluator \
        --input-file-path "${input_file_path}" \
        --target-file-path "${target_file_path}" \
        --config_file "${config_file}" \
        --print-all-scores
fi
