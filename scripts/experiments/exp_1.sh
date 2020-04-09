######### EXPERIENCES #########

### Experience 1: 

# 1. Train a back-translater model
python -m src.train_transformer --cfg_path config_files/transformer_back_cfg.json

# 2. Use the back translater model to generate synthetic data from the monolingual corpora
python -m src.generate_synthetic -i data/no_punctuation/unaligned.en -p  -c config_files/transformer_eval_cfg.json -n 100000

# 3. Concatenate generated synthetic data with the exisiting parallel corpus
./scripts/concatenate_files.sh data/synthetic 100000 data/synthetic

# 4. Train a translater model on the augmented data set
python -m src.train_transformer --cfg_path config_files/transformer_cfg.json

# 5. Evaluate the translater model
python -m src.evaluator --target-file-path path_to_target_file --input-file-path path_to_input_file