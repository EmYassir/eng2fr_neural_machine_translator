"""
Utility functions for Transformer model
"""

from typing import Tuple, Dict

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


def load_transformer(
        config: Dict,
        tokenizer_source: tfds.features.text.SubwordTextEncoder,
        tokenizer_target: tfds.features.text.SubwordTextEncoder
) -> Transformer:
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


def evaluate(inp_sentence: str, tokenizer_source: tfds.features.text.SubwordTextEncoder,
             tokenizer_target: tfds.features.text.SubwordTextEncoder, max_length_pred: int,
             transformer: Transformer) -> Tuple:
    """
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

    for i in range(max_length_pred):
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


# Normalizing function: transforms output scores into logs of softmax
def log_prob_from_logits(logits, reduce_axis=-1):
    return logits - tf.reduce_logsumexp(logits, axis=reduce_axis, keepdims=True)


# TODO : fix output (not Tuple, but tensor (just write Tensor? EagerTensor?))
# modified evaluate() function to implement beam_search for single sentence translation
def evaluate_beam(inp_sentence: str, tokenizer_source: tfds.features.text.SubwordTextEncoder,
                  tokenizer_target: tfds.features.text.SubwordTextEncoder, max_length_pred: int,
                  transformer: Transformer, beam_size: int, alpha=0.6) -> Tuple:
    """
    Takes an input sentence and generate sequences of tokens for its translation
    using beam_search
    :param inp_sentence: Input sentence in source language
    :param tokenizer_source: Tokenizer for source language
    :param tokenizer_target: Tokenizer for target language
    :param max_length_pred: Maximum length of output sequence
    :param transformer: Trained Transformer model
    :param beam_size: size of search beam
    :param alpha: factor to set sequence length penalty
    :return: The sequence of token ids in target language, the attention weights
    """
    # Length penalty applied to longer sequence scores = (5+len(output_sequence)/6) ^ alpha
    # length penalty is neutralized when alpha = 0.0; Wu et al 2016 suggest alpha = [0.6-0.7]
    alpha = alpha
    # Initialized to 1 =  no length penalty
    length_penalty = 1.0

    start_token = [tokenizer_source.vocab_size]
    end_token = [tokenizer_source.vocab_size + 1]

    # Adding the start and end token to input
    inp_sentence = start_token + tokenizer_source.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # The first word to the transformer should be the target start token
    decoder_input = [tokenizer_target.vocab_size]
    deco_input = tf.expand_dims(decoder_input, 0)

    # Initialize log (probability) of sequence score to 0.0 (prob_sequence = 1.0)
    # key (string) = step identifier ; value (tuple) = (output sequence, score, flag)
    # where score = log(prob_sequence), and flag = 1 if sequence is finished (end_token outputed)
    candidates = {
        'Step0': (deco_input, 0.0, 0)}

    for i in range(max_length_pred):
        #  Set length penalty to adjust score based on increasing sequence length
        old_length_penalty = length_penalty
        length_penalty = ((5+i+1)/6)**alpha  # (5+len(decode)/6) ^ -\alpha

        # store best results; add k results per candidate, then keep top k (across candidates)
        next_candidates = {}
        candi_num = 0

        # Loop on each retained candidate in dictionary
        for label, candidate in candidates.items():
            candi_num += 1
            sequence = candidate[0]
            score = candidate[1]
            flag = candidate[2]
            # if sequence is finished (flag = 1) and it contains at least one token
            if flag == 1:
                # TODO : determine if this is necessary
                if sequence.shape[-1] > 1:
                    # copy key-value pair as-is into next_candidates dictionary
                    next_candidates[label] = candidate
            # otherwise expand sequence
            else:
                enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, sequence)
                # predictions.shape == (batch_size, seq_len, vocab_size)
                predictions, attention_weights = transformer(encoder_input,
                                                             sequence,
                                                             False,
                                                             enc_padding_mask,
                                                             combined_mask,
                                                             dec_padding_mask)

                # select the last word from the seq_len dimension
                predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

                # Normalize output: transform predictions to log of softmax
                norm_predict = log_prob_from_logits(predictions)

                # select top_k values among output words
                predicted_values, predicted_ids = tf.nn.top_k(norm_predict, k=beam_size)
                # predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

                # compute score and build sequence for each top word
                for k in range(beam_size):
                    pred_val = tf.cast(predicted_values[:, :, k], tf.float32)
                    pred_id = tf.cast(predicted_ids[:, :, k], tf.int32)

                    # Create unique key for dictionary entry
                    entry_label = 'Step'+str(i+1)+'_'+str(candi_num)+str(k+1)

                    # Verify if sequence is finished (predicted_id is equal to target end token)
                    if pred_id == tokenizer_target.vocab_size + 1:
                        # preserve sequence and score, but set "finished" flag to 1
                        next_candidates[entry_label] = (sequence, score, 1)

                    # Calculate new score, generate new sequence, and save in dictionary
                    else:
                        new_score = (score*old_length_penalty + pred_val)/length_penalty
                        new_seq = tf.concat([sequence, pred_id], axis=-1)
                        next_candidates[entry_label] = (new_seq, new_score, 0)

                        # To help debug
                        # pred_sentence = tokenizer_target.decode([i for i in tf.squeeze(new_seq, axis=0)
                        #                                        if i < tokenizer_target.vocab_size])
                        # print(entry_label, ' : ', pred_sentence)

        # Reduce number of next candidates to beam_size
        # compare scores and identify top k
        allscores = []
        for key, value in next_candidates.items():
            allscores.append((value[1], key))
        top_scores = sorted(allscores, key=lambda tup: tup[0], reverse=True)[:beam_size]

        # update dictionary of candidates for next iteration, size = beam_size
        candidates = {}
        sum_done = 0
        for score in top_scores:
            candidates[score[1]] = next_candidates[score[1]]
            sum_done += candidates[score[1]][2]

        # stop loop before max_length_pred if all top sequences are completed
        if sum_done == beam_size:
            break
    # Select top sequence within dictionary, and return as squeezed output seq
    top_scores = []
    for key, value in candidates.items():
        top_scores.append((value[1], key))

    best_sequence = candidates[sorted(top_scores, key=lambda tup: tup[0], reverse=True)[0][1]][0]

    return tf.squeeze(best_sequence, axis=0)


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
              transformer: Transformer, beam_size=None, alpha=0.6, plot: str = "") -> str:
    """
    Translate a sentence from source to target language
    :param inp_sentence: input sentence in source language
    :param tokenizer_source: tokenizer for source language
    :param tokenizer_target: tokenizer for target language
    :param max_length_pred: maximum number of tokens in output sentence
    :param transformer: Trained Transformer model
    :param beam_size: size of beam search to generate translation (None = use gready search)
    :param alpha: factor to set sequence length penalty for beam search
    :param plot: Name of layer to plot (will not plot if "")
    :return: The translated sentence in target language
    """
    if beam_size is None:
        result, attention_weights = evaluate(inp_sentence, tokenizer_source, tokenizer_target,
                                             max_length_pred, transformer)
        predicted_sentence = tokenizer_target.decode([i for i in result if i < tokenizer_target.vocab_size])
        if plot:
            plot_attention_weights(attention_weights, inp_sentence, result, plot, tokenizer_source, tokenizer_target)

    else:
        result = evaluate_beam(inp_sentence, tokenizer_source, tokenizer_target, max_length_pred,
                               transformer, beam_size, alpha)
        predicted_sentence = tokenizer_target.decode([i for i in result if i < tokenizer_target.vocab_size])

    return predicted_sentence
