"""
Utility functions to manipulate tokenizers and word2vec embeddings
"""

from gensim.models import KeyedVectors
import tensorflow_datasets as tfds
import numpy as np


def token_to_subword(token: int, tokenizer: tfds.features.text.SubwordTextEncoder) -> str:
    """
    Give subword corresponding to token id
    :param token: Token (index)
    :param tokenizer: Tokenizer
    :return subword corresponding to the token
    """
    subword = tokenizer._id_to_subword(token - 1)
    # OOV words are broken down into bytes, ignore those
    if isinstance(subword, bytes):
        subword = None
    return subword


def break_file_into_subwords(input_file_path: str, output_file_path: str,
                             tokenizer: tfds.features.text.SubwordTextEncoder) -> None:
    with open(input_file_path) as in_f:
        with open(output_file_path, "w") as out_f:
            for line in in_f.readlines():
                line = line.strip()
                encoded_line = tokenizer.encode(line)
                subwords = [token_to_subword(token, tokenizer) for token in encoded_line]
                valid_subwords = [subword for subword in subwords if subword is not None]
                subwords_sentence = " ".join(valid_subwords)
                out_f.write(subwords_sentence + "\n")


def get_pretrained_weights(embedding_weights: np.ndarray,
                           tokenizer: tfds.features.text.SubwordTextEncoder,
                           word2vec_model_path: str):
    word2vec_model = KeyedVectors.load(word2vec_model_path)
    # the last 2 weights are for start/end tokens
    for i in range(1, tokenizer.vocab_size):
        subword = token_to_subword(i, tokenizer)
        if subword is not None and subword in word2vec_model.vocab:
            subword_vector = word2vec_model[subword]
            embedding_weights[i] = subword_vector
    return embedding_weights
