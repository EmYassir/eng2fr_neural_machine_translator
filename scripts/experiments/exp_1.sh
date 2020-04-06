######### EXPERIENCES #########

### Experience 1: 

# 1. Train a back-translater model
python -m src.train_transformer --cfg_path config_files/transformer_back_cfg.json

# 2. Use the back translater model to generate synthetic data from the monolingual corpora

# 3. Concatenate generated synthetic data with the exisiting parallel corpus

# 4. Train a translater model on the augmented data set
python -m src.train_transformer --cfg_path config_files/transformer_cfg.json

# 5. Use the translater to generate data 

# 6. Compute the bleu score