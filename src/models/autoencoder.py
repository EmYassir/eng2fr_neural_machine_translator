from src.models.Transformer import Transformer
from src.utils.transformer_utils import load_transformer



def evaluate(encoder_input: tf.Tensor,
             tokenizer_target: tfds.features.text.SubwordTextEncoder,
             transformer: Transformer) -> tf.Tensor:
    """
    Takes encoded input sentence ands generate the sequence of tokens for its translation
    :param encoder_input: Encoded input sentences
    :param tokenizer_target: Tokenizer for target language
    :param transformer: Trained Transformer model
    :return: Output sentences encoded for target language
    """
    # The first word to the transformer should be the target start token
    decoder_input = [tokenizer_target.vocab_size] * encoder_input.shape[0]
    output = tf.reshape(decoder_input, (-1, 1))
    # TODO Consider if there is a better heuristic
    #  The higher max_additional_tokens, the longer it takes to evaluate. On the other side, a lower number risks
    #  returning incomplete sentences.
    max_additional_tokens = int(0.5 * encoder_input.shape[1])
    max_length_pred = encoder_input.shape[1] + max_additional_tokens

    for _ in range(max_length_pred):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, _ = transformer(encoder_input,
                                     output,
                                     False,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # concatenate the predicted_id to the output which is given to the decoder as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    # TODO re-add attention weights as output if we want to print them
    return output


class AutoEncoder:
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        
        self.encoder = Transformer(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Transformer(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)


    def load_encoder_decoder(config_encoder, config_decoder):
        # Encoder
        tokenizer_source_path = os.path.join(config_encoder.saved_path, config_encoder["tokenizer_source_path"])
        tokenizer_target_path = os.path.join(config_encoder.saved_path, config_encoder["tokenizer_target_path"])
        checkpoint_path_best = os.path.join(config_encoder.saved_path, config_encoder["checkpoint_path_best"])
        translation_batch_size = config_encoder["translation_batch_size"]
        tokenizer_source = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_source_path)
        tokenizer_target = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_target_path)
        self.encoder = load_transformer(config_encoder, tokenizer_source, tokenizer_target)
        
         # Decoder
        tokenizer_source_path = os.path.join(config_encoder.saved_path, config_encoder["tokenizer_source_path"])
        tokenizer_target_path = os.path.join(config_encoder.saved_path, config_encoder["tokenizer_target_path"])
        checkpoint_path_best = os.path.join(config_encoder.saved_path, config_encoder["checkpoint_path_best"])
        translation_batch_size = config_encoder["translation_batch_size"]
        tokenizer_source = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_source_path)
        tokenizer_target = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_target_path)
        self.decoder = load_transformer(config_decoder, tokenizer_source, tokenizer_target)
        
        
    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask)  # (batch_size, tar_seq_len, target_vocab_size)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  

        return final_output, attention_weights
