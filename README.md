# ift6759_project2
Low-ressource machine transaltion system for IFT6759 course

## Training the Transformer
From the project root folder, type the following command
```bash
python src/train_transformer.py --cfg_path config_files/transformer_cfg.json
```

## Evaluate the model
The generate predictions is currently configured for the transformer model. Make sure the config file path is up
to date and load the necessary parameters
```bash
python src/evaluator.py --target-file-path path_to_target_file --input-file-path path_to_input_file
```