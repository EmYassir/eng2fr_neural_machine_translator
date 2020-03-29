"""
Utility functions to manipulate data
"""

from typing import List, Optional, Tuple
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds


def project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent.parent


def preprocess_sentence(sentence: str) -> str:
    """
    Add <start> and <stop> tokens to sentence
    :param sentence: sentence to process
    :return: sentence with tokens
    """
    sentence = '<start> ' + sentence + ' <end>'
    sentence = sentence.rstrip().strip()
    return sentence


def max_length(tensor):
    """
    :return max length of tensor in a batch
    :param tensor: minibatch of tensors
    :return: the length of the longest tensor
    """
    return max(len(t) for t in tensor)


def create_dataset(source: str, target: Optional[str], max_seq_length: Optional[int],
                   num_examples: Optional[int]) -> Tuple:
    """
    Takes a source and target file and return two lists of paired data points
    :param source: path to source file
    :param target: path to target file
    :param max_seq_length: max accepted length for sequence
    :param num_examples: max number of examples, take all if None
    :return: list of source sentece, list of target sentences
    """
    with open(source, encoding="UTF-8") as source_file:
        source_lines = source_file.read().strip().split('\n')
    if target is not None:
        with open(target, encoding="UTF-8") as target_file:
            target_lines = target_file.read().strip().split('\n')
        assert len(source_lines) == len(target_lines)
    source_data = []
    target_data = []
    for i in range(len(source_lines[:num_examples])):
        if max_seq_length is None:
            source_data.append(preprocess_sentence(source_lines[i]))
            if target is not None:
                target_data.append(preprocess_sentence(target_lines[i]))
        elif target is None and len(source_lines[i].split()) <= max_seq_length:
            source_data.append(preprocess_sentence(source_lines[i]))
        elif len(source_lines[i].split()) <= max_seq_length and len(target_lines[i].split()) <= max_seq_length:
            source_data.append(preprocess_sentence(source_lines[i]))
            target_data.append(preprocess_sentence(target_lines[i]))
    return source_data, target_data


def create_transformer_dataset(source: str, target: Optional[str],
                               num_examples: Optional[int]) -> tf.data.Dataset:
    """
    Takes a source and target file and return a dataset
    :param source: path to source file
    :param target: path to target file
    :param num_examples: max number of examples, take all if None
    :return: tf Dataset object
    """
    with open(source, encoding="UTF-8") as source_file:
        source_lines = source_file.readlines()
    if target is not None:
        with open(target, encoding="UTF-8") as target_file:
            target_lines = target_file.readlines()
        assert len(source_lines) == len(target_lines)
    source_data = []
    target_data = []
    for source_line in source_lines[:num_examples]:
        source_data.append(source_line.strip())
    if target is not None:
        for target_line in target_lines[:num_examples]:
            target_data.append(target_line.strip())
    else:
        target_data = [""] * len(source_lines)
    dataset = tf.data.Dataset.from_tensor_slices((source_data, target_data))
    return dataset


def tokenize(lang: List[str], lang_model: Optional[object]) -> Tuple:
    """
    Transforms a list of sentence into a list of list of indexes the correspond to their index in the
    language model with a +1 offset
    :param lang: a list of sentences
    :param lang_model: a language model (ex: Word2Vec from gensim
    :return: A list of list of indexes, a tokenizer if no language model was given
    """
    # Use language model if supplied
    if lang_model is not None:
        lang_tokenizer = None
        tensor = []
        for line in lang:
            tokens = []
            for word in line.split():
                # transform word to its index (with +1 offset) in the language model
                if word in lang_model.vocab.keys():
                    tokens.append(lang_model.vocab[word].index + 1)
                else:  # treat word as unknown if not in language model
                    tokens.append(lang_model.vocab["<unk>"].index + 1)
            tensor.append(tokens)
    else:
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        lang_tokenizer.fit_on_texts(lang)
        tensor = lang_tokenizer.texts_to_sequences(lang)

    # pad with zeros to ensure all sequences have same length
    # TODO see if this can be done later to reduce the size of some minibatches
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


def load_dataset(input_path: str, target_path: Optional[str], max_seq_length: Optional[int] = None,
                 num_examples: Optional[int] = None, inp_lang_model: Optional[object] = None,
                 targ_lang_model: Optional[object] = None) -> Tuple:
    """
    Load a dataset by creating list of indexes from source and target files data
    :param input_path: path to training.lang1 file
    :param target_path: path to trainging.lang2 file
    :param max_seq_length: maximum length of sequence allowed (use only for training)
    :param num_examples: maximum of examples used, takes all if None
    :param inp_lang_model: input language model (ex: Word2Vec from gensim)
    :param targ_lang_model: target language model (ex: Word2Vec from gensim)
    :return: The input indexes sequences, the target indexes sequences, the input tokenizer and the target tokenizer
    """
    inp_lang, targ_lang = create_dataset(input_path, target_path, max_seq_length, num_examples)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang, inp_lang_model)
    if target_path is not None:
        target_tensor, targ_lang_tokenizer = tokenize(targ_lang, targ_lang_model)
    else:
        target_tensor = []
        targ_lang_tokenizer = None
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def build_tokenizer(files_list, target_vocab_size=2 ** 13):
    language = []
    for file in files_list:
        with open(file, "r", encoding="utf-8") as lang_file:
            language += lang_file.readlines()
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (line for line in language), target_vocab_size=target_vocab_size)
    return tokenizer
