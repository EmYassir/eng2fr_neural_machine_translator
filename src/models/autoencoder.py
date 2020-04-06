import numpy as np
import tensorflow as tf
from src.models.Transformer import Transformer
from src.utils.transformer_utils import load_transformer



class AutoEncoder(tf.keras.Model):
    
    def __init__(self, config, config_enc, config_dec, tokenizer_source, tokenizer_target):
        super(AutoEncoder, self).__init__()
        # Aligning batch size
        config_enc_copy = config_enc.copy()
        config_enc_copy["translation_batch_size"] = config["batch_size"]
        config_dec_copy = config_dec.copy()
        config_dec_copy["translation_batch_size"] = config["batch_size"]
        # Loading transformers
        self.encoder = load_transformer(config_enc_copy, tokenizer_source, tokenizer_target)
        self.decoder = load_transformer(config_dec_copy, tokenizer_target, tokenizer_source)

        
    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask)  # (batch_size, tar_seq_len, target_vocab_size)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  

        return final_output, attention_weights
