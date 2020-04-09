### Experience 4: 

# 1. Train a back-translater model on the parallel data set
python -m src.train_transformer --cfg_path config_files/transformer_back_cfg.json

# 2. Train a translater model on the parallel data set
python -m src.train_transformer --cfg_path config_files/transformer_cfg.json

# 3. Train the autoencoder model tgt->tgt on the monolingual corpora

# 4. Evaluate the translater model

# 5. Train the autoencoder model src->src on the monolingual corpora

# 6. Evaluate the translater model