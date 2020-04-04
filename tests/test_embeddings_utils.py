import json
import os

import numpy as np
import pytest
import tensorflow_datasets as tfds
from gensim.models import KeyedVectors

from src.utils.data_utils import project_root
from src.utils.embeddings_utils import token_to_subword
from src.utils.transformer_utils import load_transformer

source_lang_model_path = os.path.join(project_root(), "src", "embedding_models",
                                      "word2vec", "english_w2v_subwords_256.bin")
target_lang_model_path = os.path.join(project_root(), "src", "embedding_models",
                                      "word2vec", "french_w2v_subwords_256.bin")


@pytest.fixture
def source_lang_model():
    source_lang_model = KeyedVectors.load(source_lang_model_path, mmap='r')
    return source_lang_model


@pytest.fixture
def target_lang_model():
    target_lang_model = KeyedVectors.load(target_lang_model_path, mmap='r')
    return target_lang_model


@pytest.fixture
def source_tokenizer():
    source_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(os.path.join(project_root(),
                                                                                         "tokenizer",
                                                                                         "tokenizer_en.save"))
    return source_tokenizer


@pytest.fixture
def target_tokenizer():
    target_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(os.path.join(project_root(),
                                                                                         "tokenizer",
                                                                                         "tokenizer_fr.save"))
    return target_tokenizer


def test_get_pretrained_weights(source_tokenizer, target_tokenizer, source_lang_model, target_lang_model):
    with open(os.path.join(project_root(), "config_files", "debug", "transformer_cfg.json")) as f_config:
        config = json.load(f_config)
    transformer = load_transformer(config,
                                   source_tokenizer,
                                   target_tokenizer,
                                   source_lang_model_path,
                                   target_lang_model_path)
    line = "This a test line"
    encoded_line = source_tokenizer.encode(line)
    subwords = [token_to_subword(token, source_tokenizer) for token in encoded_line]
    for i, subword in enumerate(subwords):
        if subword is not None:
            token = encoded_line[i]
            transformer_embedding = transformer.encoder.embedding(token)
            w2v_embedding = source_lang_model[subword]
            np.testing.assert_array_equal(transformer_embedding, w2v_embedding)
