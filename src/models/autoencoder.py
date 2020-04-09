import tensorflow as tf
from src.utils.transformer_utils import load_transformer
from src.utils.transformer_utils import translate_batch
from src.utils.transformer_utils import create_masks


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

    def call(self, inp, tar):
        # tokenizers
        tokenizer_source = self.tokenizer_src
        tokenizer_target = self.tokenizer_tgt

        def encode(source, target):
            # Add start and end token
            source_tokenized = [tokenizer_source.vocab_size] + tokenizer_source.encode(
                source.numpy()) + [tokenizer_source.vocab_size + 1]

            target_tokenized = [tokenizer_target.vocab_size] + tokenizer_target.encode(
                target.numpy()) + [tokenizer_target.vocab_size + 1]

            return source_tokenized, target_tokenized

        def tf_encode(source, target):
            # encapsulate our encode function in a tf functions so it can be called on tf tensor
            result_source, result_target = tf.py_function(encode, [source, target], [tf.int64, tf.int64])
            result_source.set_shape([None])
            result_target.set_shape([None])
            return result_source, result_target

        # Getting translations
        tf.print("type of %d", inp.dtype)
        translations = translate_batch(self.encoder, tokenizer_source, tokenizer_target, inp.numpy(), self.batch_size, True)
        tf.print(f'translations size == {len(translations)}')
        tf.print(translations)
        # Converting translations to tf dataset to feed them to decoder
        dataset = tf.data.Dataset.from_tensor_slices((translations, tar))
        data_preprocessed = (
            # cache the dataset to memory to get a speedup while reading from it.
            dataset.map(tf_encode).cache()
        )
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(data_preprocessed, tar)
        preds, _ = self.decoder(data_preprocessed, tar, True, enc_padding_mask, combined_mask, dec_padding_mask)
        return preds
