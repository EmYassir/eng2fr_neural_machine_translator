# ift6759_project2
Low-ressource machine translation system for IFT6759 course

## Run the evaluator.py on the test set
Use the bash script in `scripts/evaluator.sh` to run the evaluation tests. The script
uses the virtual environment in our submission team folder (at `/project/cq-training-1/project2/submissions/team09/venv/` on the cluster).
The script then submits a job to helios to run the evaluation on a compute-node. To run the evaluation script:
```bash
sbatch scripts/evaluator.sh [input-file-path] [target-file-path] [print-all-scores]
```
Make sure to call the script from inside /project/cq-training-1/project2/submissions/team09/ift6759_project2

Also make sure that the [input-file-path] and [target-file-path] are the *absolute* path and not the relative path
to the current folder from where the script is ran. The third argument [print-all-scores] is an optional flag and will
default to false if not set.
For example, we ran the command
```bash
sbatch scripts/evaluator.sh /project/cq-training-1/project2/submissions/team09/data/validation.lang1 /project/cq-training-1/project2/submissions/team09/data/validation.lang2
```
to make sure that the evaluator script was working, where the `validation.lang` files contain 1k parallel sentences.

TROUBLESHOOTING:
Make sure that our model is loaded, you should see the line: 
Latest checkpoint restored from  /project/cq-training-1/project2/submissions/team09/ift6759_project2/best_models/second_iteration_forward
in your slurm file. If not, try running the code again, making sure you are calling the script from the correct directory.

If you get a OOM error, try lowering the translation_batch_size parameter in the config file at:
/project/cq-training-1/project2/submissions/team09/ift6759_project2/config_files/transformer_eval_cfg.json
batch size of 48 worked fine on our validation set but longer sequences or more samples might necessitate a lower batch size. 
