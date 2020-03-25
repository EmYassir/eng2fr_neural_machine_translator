from src.utils.data_utils import load_dataset
from gensim.models import KeyedVectors
import numpy as np
import pytest


@pytest.fixture
def targ_lang_model():
    targ_lang_model = KeyedVectors.load("src/embedding_models/word2vec/french_w2v_200.bin", mmap='r')
    # Add unknown token to target vocabulary with mean vector as value
    targ_mean_vector = np.mean(targ_lang_model[list(targ_lang_model.vocab.keys())], axis=0)
    targ_lang_model.add(["<unk>"], [targ_mean_vector])
    return targ_lang_model


def test_load_dataset(targ_lang_model):
    inp_tensor, targ_tensor, inp_lang_tokenizer, targ_lang_tokenizer = load_dataset("tests/train_test.lang1",
                                                                                    "tests/train_test.lang2",
                                                                                    max_seq_length=None,
                                                                                    num_examples=None,
                                                                                    targ_lang_model=targ_lang_model)
    assert len(inp_tensor) == 2
    assert len(inp_tensor) == len(targ_tensor)


def test_load_dataset_num_examples(targ_lang_model):
    inp_tensor, targ_tensor, inp_lang_tokenizer, targ_lang_tokenizer = load_dataset("tests/train_test.lang1",
                                                                                    "tests/train_test.lang2",
                                                                                    max_seq_length=None,
                                                                                    num_examples=1,
                                                                                    targ_lang_model=targ_lang_model)
    assert len(inp_tensor) == 1
    assert len(inp_tensor) == len(targ_tensor)


def test_load_dataset_max_seq_length(targ_lang_model):
    inp_tensor, targ_tensor, inp_lang_tokenizer, targ_lang_tokenizer = load_dataset("tests/train_test.lang1",
                                                                                    "tests/train_test.lang2",
                                                                                    max_seq_length=20,
                                                                                    num_examples=None,
                                                                                    targ_lang_model=targ_lang_model)
    assert len(inp_tensor) == 1
    assert len(inp_tensor) == len(targ_tensor)


def test_load_dataset_test(targ_lang_model):
    inp_tensor, targ_tensor, inp_lang_tokenizer, targ_lang_tokenizer = load_dataset("tests/train_test.lang1",
                                                                                    None,
                                                                                    max_seq_length=None,
                                                                                    num_examples=None,
                                                                                    targ_lang_model=None)
    assert len(inp_tensor) == 2
    assert len(targ_tensor) == 0
