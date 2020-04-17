import tensorflow as tf
from src.utils.transformer_utils import load_transformer
from src.utils.transformer_utils import create_masks
from src.utils.transformer_utils import softargmax


class AutoEncoder(tf.keras.Model):

    def __init__(self, config, config_enc, config_dec, tokenizer_source, tokenizer_target):
        super(AutoEncoder, self).__init__()
        # Batch size
        self.batch_size = config["batch_size"]
        # Tokenizers
        self.tokenizer_src = tokenizer_source
        self.tokenizer_tgt = tokenizer_target
        # Loading transformers
        self.encoder = load_transformer(config_enc, tokenizer_source, tokenizer_target)
        self.decoder = load_transformer(config_dec, tokenizer_target, tokenizer_source)

    def call(self, inp, tar, tar_inp, tar_real):
        # Masks
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        # Getting translations
        intermediate_logits, w1 = self.encoder(inp, tar_inp, True, enc_padding_mask,
                                               combined_mask, dec_padding_mask)
        # Differentiable argmax for backprop?
        intermediate_predictions = softargmax(intermediate_logits)

        # Masks
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(intermediate_predictions, tar_real)
        predictions, w2 = self.decoder(intermediate_predictions, tar_real, True,
                                       enc_padding_mask, combined_mask, dec_padding_mask)
        return predictions, w1, w2
