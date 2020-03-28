# ift6759_project2
Low-ressource machine transaltion system for IFT6759 course

## Run the evaluator.py on the test set
We use the bash script in `scripts/evaluator.sh` to run the evaluation tests. The script
uses the virtual environment in our team folder (at `/project/cq-training-1/project2/teams/team09/venv/` on the cluster).
The script then submits a job to helios to run the evaluation on a compute-node. To run the evaluation script:
```bash
sbatch scripts/evaluator.sh [input-file-path] [target-file-path] [print-all-scores]
```
Make sure that the [input-file-path] and [target-file-path] are the *absolute* path and not the relative path
to the current folder from where the script is ran. The third argument [print-all-scores] is an optional flag and will 
default to false if not set.
For example, we ran the command
```bash
sbatch scripts/evaluator.sh /project/cq-training-1/project2/teams/team09/data/validation.lang1 /project/cq-training-1/project2/teams/team09/data/validation.lang2
```
to make sure that the evaluator script was working, where the `validation.lang` files contain 1k parallel sentences.

## Training the Transformer
From the project root folder, type the following command
```bash
python src/train_transformer.py --cfg_path config_files/transformer_cfg.json
```
Note, the content of the data folder should be the same as the data folder of the shared team directory on Helios

## Evaluate the model
The generate predictions is currently configured for the transformer model. Make sure the config file path is up
to date and load the necessary parameters
```bash
python src/evaluator.py --target-file-path path_to_target_file --input-file-path path_to_input_file
```