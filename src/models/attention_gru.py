"""
Machine translation model that generate prediction using GRU with attention
"""

from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding


class Encoder(tf.keras.Model):
    """
    Part of the model that takes a sequence of indexes and return a sequence of embeddings using GRU for context
    """
    def __init__(self, vocab_size: int, embedding_dim: int, enc_units: int, batch_size: int,
                 lang_model: Optional[object] = None) -> None:
        """
        Initialize the encoder
        :param vocab_size: size of input language vocabulary
        :param embedding_dim: size of embedding
        :param enc_units: size of hidden state and outputs
        :param batch_size: batch size
        :param lang_model: language model (ex: gensim Word2Vec)
        """
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        # Use pre-trained embeddings if a language model is given
        if lang_model is not None:
            embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))
            for word in lang_model.vocab.keys():
                embedding_vector = lang_model[word]
                word_idx = lang_model.vocab[word].index
                embedding_matrix[word_idx + 1] = embedding_vector
            self.embedding = Embedding(vocab_size + 1,
                                       embedding_dim,
                                       weights=[embedding_matrix],
                                       mask_zero=True,
                                       trainable=True)
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        """
        Generate a sequence of output and a final hidden state
        :param x: sequences of input indexes (batch_size, sequence length, 1)
        :param hidden: hidden state (batch_size, units)
        :return: sequence of output (batch_size, sequence length, units)
        """
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        """
        Create initial hidden_state of size (batch_size, units)
        :return: the initial hidden state tensor
        """
        return tf.zeros((self.batch_size, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    """
    Attention module that generate a context vector from the decoder hidden state and the encoder output
    """
    def __init__(self, units):
        """
        Create weight vectors
        :param units:
        """
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, units)
        # query_with_time_axis shape == (batch_size, 1, units)
        # values shape == (batch_size, max_len, units)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    """
    Part of the model that tries to predict the next word of the output using the current input and the context
    given by the attention module
    """
    def __init__(self, vocab_size: int, embedding_dim: int, dec_units: int, batch_size: int,
                 lang_model: Optional[object]) -> object:
        """
        Initialize decoder
        :param vocab_size: size of target language vocabulary
        :param embedding_dim: size of embedding
        :param dec_units: size of hidden states
        :param batch_size: batch size
        :param lang_model: language model (ex: gensim Word2Vec)
        """
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        if lang_model is not None:
            embedding_matrix = np.zeros((vocab_size+1, embedding_dim))
            for word in lang_model.vocab.keys():
                embedding_vector = lang_model[word]
                word_idx = lang_model.vocab[word].index
                embedding_matrix[word_idx + 1] = embedding_vector
            self.embedding = Embedding(vocab_size+1,
                                       embedding_dim,
                                       weights=[embedding_matrix],
                                       mask_zero=True,
                                       trainable=True)
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size + 1)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        """
        Output the prediction for the next word from encoder output, input and hidden state
        :param x: The index of the last known word
        :param hidden: The hidden state (first from the encoder, then last decoder step)
        :param enc_output: The embeddings created by the encoder
        :return: The logits to compute probabilities of next word
        """
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights
