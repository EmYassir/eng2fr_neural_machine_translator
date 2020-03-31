"""
Utility functions for Transformer model
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from src.models.Transformer import Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Creates custom learning rate scheduler for Transformer
    """
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def load_transformer(config: Dict,
                     tokenizer_source: tfds.features.text.SubwordTextEncoder,
                     tokenizer_target: tfds.features.text.SubwordTextEncoder) -> Transformer:
    """
    Load transformer model
    :param config: path to config file
    :param tokenizer_source: Source language tokenizer
    :param tokenizer_target: Target language tokenizer
    :return: A transformer model created from parameters in config
    """
    # Set hyperparameters
    num_layers = config["num_layers"]
    d_model = config["d_model"]
    dff = config["dff"]
    num_heads = config["num_heads"]
    dropout_rate = config["dropout_rate"]

    source_vocab_size = tokenizer_source.vocab_size + 2
    target_vocab_size = tokenizer_target.vocab_size + 2
    tf.print(f"Source_vocab_size = {source_vocab_size} and Target_vocab_size = {target_vocab_size}")

    transformer = Transformer(num_layers, d_model, num_heads, dff,
                              source_vocab_size, target_vocab_size,
                              pe_input=source_vocab_size,
                              pe_target=target_vocab_size,
                              rate=dropout_rate)
    return transformer


def create_padding_mask(seq):
    """
    Create mask to use padding provided by input sequence
    :param seq: Input sequence
    :return: Mask that masks elements where input sequence is 0
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    """
    Create mask to prevent looking at future elements in decoder
    :param size: size of sequence
    :return: Mask that masks future elements in decoder input sequence
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    """
    Create masks for transformer
    :param inp: input sequence
    :param tar: target sequence
    :return: encoder padding mask, combined_mask and decoder padding mask
    """
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def evaluate_old(inp_sentence: str, tokenizer_source: tfds.features.text.SubwordTextEncoder,
                 tokenizer_target: tfds.features.text.SubwordTextEncoder, max_length_pred: int,
                 transformer: Transformer) -> Tuple:
    """
    Deprecated, used only by the translate method that don't use batches

    Takes an input sentence and generate the sequence of tokens for its translation
    :param inp_sentence: Input sentence in source language
    :param tokenizer_source: Tokenizer for source language
    :param tokenizer_target: Tokenizer for target language
    :param max_length_pred: Maximum length of output sequence
    :param transformer: Trained Transformer model
    :return: The sequence of token ids in target language, the attention weights
    """
    start_token = [tokenizer_source.vocab_size]
    end_token = [tokenizer_source.vocab_size + 1]

    # Adding the start and end token to input
    inp_sentence = start_token + tokenizer_source.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # The first word to the transformer should be the target start token
    decoder_input = [tokenizer_target.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for _ in range(max_length_pred):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer_target.vocab_size + 1:
            return tf.squeeze(output, axis=0), attention_weights

        # concatenate the predicted_id to the output which is given to the decoder as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


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


def plot_attention_weights(attention, sentence, result, layer, tokenizer_source, tokenizer_target):
    """
    Plot attention weights for a given layer
    :param attention: Attention weights
    :param sentence: Tokenized input sentence
    :param result: Tokenized translated sentence
    :param layer: Name of layer to plot ex('decoder_layer4_block2')
    :param tokenizer_source: Source language tokenizer
    :param tokenizer_target: Target language tokenizer
    """
    fig = plt.figure(figsize=(16, 8))

    sentence = tokenizer_source.encode(sentence)

    attention = tf.squeeze(attention[layer], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(sentence) + 2))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result) - 1.5, -0.5)

        ax.set_xticklabels(
            ['<start>'] + [tokenizer_source.decode([i]) for i in sentence] + ['<end>'],
            fontdict=fontdict, rotation=90)

        ax.set_yticklabels([tokenizer_target.decode([i]) for i in result
                            if i < tokenizer_target.vocab_size],
                           fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1))

    plt.tight_layout()
    plt.show()


def translate(inp_sentence: str, tokenizer_source: tfds.features.text.SubwordTextEncoder,
              tokenizer_target: tfds.features.text.SubwordTextEncoder, max_length_pred: int,
              transformer: Transformer, plot: str = "") -> str:
    """
    Deprecated, still works for a single example and can plot attention

    Translate a sentence from source to target language
    :param inp_sentence: input sentence in source language
    :param tokenizer_source: tokenizer for source language
    :param tokenizer_target: tokenizer for target language
    :param max_length_pred: maximum number of tokens in output sentence
    :param transformer: Trained Transformer model
    :param plot: Name of layer to plot (will not plot if "")
    :return: The translated sentence in target language
    """
    result, attention_weights = evaluate_old(inp_sentence, tokenizer_source, tokenizer_target,
                                             max_length_pred, transformer)

    predicted_sentence = tokenizer_target.decode([i for i in result if i < tokenizer_target.vocab_size])
    if plot:
        plot_attention_weights(attention_weights, inp_sentence, result, plot, tokenizer_source, tokenizer_target)
    return predicted_sentence


def _get_sorted_inputs(filename: str, max_line_process: Optional[int]) -> Tuple:
    """
    Sort sentences by input lenght
    :param filename: Path to input file
    :param max_line_process: Number of line (first N lines) to keep
    :return: The sentences sorted by length and the dict to re-order them later
    """
    with open(filename) as input_file:
        records = input_file.read().split("\n")
        inputs = [record.strip() for record in records[:max_line_process]]
        if not inputs[-1]:
            inputs.pop()
    input_lens = [(i, len(line.split())) for i, line in enumerate(inputs)]
    sorted_input_lens = sorted(input_lens, key=lambda x: x[1], reverse=True)

    sorted_inputs = []
    sorted_keys = {}
    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])
        sorted_keys[index] = i
    return sorted_inputs, sorted_keys


def _encode_and_add_tokens(sentence: str, tokenizer: tfds.features.text.SubwordTextEncoder) -> List[int]:
    """
    Encode sentence and add start and end tokens
    :param sentence: Input sentence
    :param subtokenizer:
    :return:
    """
    start_token = tokenizer.vocab_size
    end_token = tokenizer.vocab_size + 1
    return [start_token] + tokenizer.encode(sentence) + [end_token]


def _trim_and_decode(ids: List[int], tokenizer: tfds.features.text.SubwordTextEncoder) -> str:
    """
    Decode list of tokens to string
    :param ids: List of tokens
    :param tokenizer: Target language tokenizer
    :return: Decoded sentence
    """
    end_token = tokenizer.vocab_size + 1
    try:
        # get index of first end token
        index = list(ids).index(end_token)
        return tokenizer.decode(ids[1:index])  # Remove start token and stop at end token
    except ValueError:  # No end token found in string
        return tokenizer.decode(ids[1:])  # Remove start token and decode all sequence


def translate_file(transformer: Transformer,
                   tokenizer_source: tfds.features.text.SubwordTextEncoder,
                   tokenizer_target: tfds.features.text.SubwordTextEncoder,
                   input_file: str,
                   batch_size: int = 32,
                   print_all_translations: bool = True,
                   max_lines_process: Optional[int] = None) -> Tuple:
    """
    Translates the sentences in input file to target language
    :param transformer: Trained Transformer model
    :param tokenizer_source: Source language tokenizer
    :param tokenizer_target: Target language tokenizer
    :param input_file: Path to input file
    :param batch_size: Batch size
    :param print_all_translations: Will print first translated sentence of every batch if True
    :param max_lines_process: Will translate only the first max_lines_process from input file
    :return: The translated sentences in a python list (sorted by input sentence length),
            The dict to re-order the sentences in the original order
    """
    # Read and sort inputs by length. Keep dictionary (original index-->new index
    # in sorted list) to write translations in the original order.
    sorted_inputs, sorted_keys = _get_sorted_inputs(input_file, max_lines_process)
    num_decode_batches = (len(sorted_inputs) - 1) // batch_size + 1

    def input_generator() -> List[int]:
        """
        Generator that yield encoded sentence from sorted inputs
        """
        for i, line in enumerate(sorted_inputs):
            if i % batch_size == 0:
                batch_num = (i // batch_size) + 1
                print(f"Decoding batch {batch_num} out of {num_decode_batches}.")
            yield _encode_and_add_tokens(line, tokenizer_source)

    def input_fn() -> tf.data.Dataset:
        """
        Create batched dataset of encoded inputs
        :return: batched dataset
        """
        dataset = tf.data.Dataset.from_generator(input_generator, tf.int64, tf.TensorShape([None]))
        dataset = dataset.padded_batch(batch_size, [None])
        return dataset

    translations = []
    for i, input_seq in enumerate(input_fn()):
        predictions = evaluate(input_seq, tokenizer_target, transformer)
        for prediction in predictions:
            translation = _trim_and_decode(prediction, tokenizer_target)
            translations.append(translation)

        if print_all_translations:
            print("Translating:")
            print(f"\tInput: {sorted_inputs[i*batch_size]}")
            print(f"\tOutput: {translations[i*batch_size]}\n")
            print("=" * 100)
    return translations, sorted_keys
