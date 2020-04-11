import tensorflow as tf
from src.utils.transformer_utils import load_transformer
from src.utils.transformer_utils import evaluate
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
    
    
    def translate(self, encoder_input, seq_len, tokenizer_target, transformer):
        # The first word to the transformer should be the target start token
        decoder_input = [tokenizer_target.vocab_size] * self.batch_size
        output = tf.reshape(decoder_input, (-1, 1))
        # Same heuristic as in Evaluate function
        #max_additional_tokens = int(0.5 * encoder_input.shape[1])
        print('@@@@@ ENCODER INPUT: ', encoder_input.get_shape()[0])
        max_additional_tokens = int(0.5 * seq_len)
        max_length_pred = encoder_input.shape[1] + max_additional_tokens
    
        for _ in range(max_length_pred):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
    
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, _ = transformer(encoder_input,
                                         output,
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)
    
            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
    
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
            # concatenate the predicted_id to the output which is given to the decoder as its input.
            output = tf.concat([output, predicted_id], axis=-1)
        return output
    


    def call(self, inp, tar):
        print('HHHHEEEEEEEEEEEEEEEEEEEEEEEEEERRRRRRRRRRRRRRRRRREEEEEEEEEE')
        vocab_size = self.tokenizer_tgt.vocab_size
        def encode(batch_tokenized):
            # Add start and end token
            batch_encoded = [vocab_size] + batch_tokenized + [vocab_size + 1]
            return batch_encoded

        def tf_encode(batch_tokenized):
            # encapsulate our encode function in a tf functions so it can be called on tf tensor
            result = tf.py_function(encode, [batch_tokenized], [tf.int64])
            #result.set_shape([None])
        tar_inp = tar[:, :-1]
        inp_real = inp[:, 1:]
        
        # Masks
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        print('############################# inp shape = ', tf.shape(inp))
        print('############################# inp size = ', tf.size(inp))
        # Getting translations
        intermediate_logits, _ = self.encoder(inp, tar, True, 
                                                   enc_padding_mask, 
                                                   combined_mask, 
                                                   dec_padding_mask)
        intermediate_predictions = tf.cast(tf.argmax(intermediate_logits, axis=-1), tf.int32)
        mini_dataset = tf.data.Dataset.from_tensors((intermediate_predictions,)) 
        tf.print('MINI DATASET ', mini_dataset)
        intermediate_predictions = (mini_dataset.map(tf_encode).cache())
        
        #intermediate_predictions = self.translate(inp, 10, self.tokenizer_tgt, self.encoder)
        print('############################# intermediate_predictions shape = ', tf.shape(intermediate_predictions))
        print('############################# intermediate_predictions size = ', tf.size(intermediate_predictions))
        #mini_dataset = tf.data.Dataset.from_tensors((intermediate_predictions, inp)) 
        # Masks
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(intermediate_predictions, inp_real)
        # predictions.shape == (batch_size, seq_len, vocab_size)
        #translated_batch = next(iter(mini_dataset))
        predictions, _ = self.decoder(intermediate_predictions, inp, True, 
                                      enc_padding_mask, combined_mask, dec_padding_mask)
        #import pdb
        #pdb.set_trace()
        print('######################## EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEENNNNNNNNNNND')
        return predictions
