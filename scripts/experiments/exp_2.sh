### Experience 3: 

# 1. Train a back-translater model
python -m src.train_transformer --cfg_path config_files/transformer_back_cfg.json

# 2. Use the back translater model to generate synthetic data from the monolingual corpora
python -m src.train_transformer --cfg_path config_files/transformer_back_cfg.json

# 3. Train a translater model on the parallel data set
python -m src.train_transformer --cfg_path config_files/transformer_cfg.json

# 4. Train the translater model on the synthetic parallel data set

# 5. Evaluate the translater model