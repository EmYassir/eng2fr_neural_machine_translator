"""train a Word2Vec model"""
from gensim.models import Word2Vec
from src.utils.embeddings_utils import break_file_into_subwords
import tensorflow_datasets as tfds


def word2vec(tokens_path: str, output: str, size: int = 100, window: int = 5, min_count: int = 5,
             workers: int = 3, sg: int = 0) -> None:
    """
    Creates Word2Vec model from a text file
    :param tokens_path: path to input text file
    :param output: path where the model should be saved
    :param size: size of the embedding
    :param window: window (context) size
    :param min_count: minimum times a word must appear to be in vocabulary
    :param workers: number of workers
    :param sg: use CBOW if 0, or Skip-Gram if 1
    """
    # load tokens from files
    sentences = []
    with open(tokens_path, 'r') as tokens_file:
        sentence = tokens_file.readline()
        while sentence:
            sentences.append(sentence.split())
            sentence = tokens_file.readline()
    # train model
    model = Word2Vec(sentences, size=size, window=window, min_count=min_count, workers=workers, sg=sg)
    # summarize the loaded model
    print(model)
    # save model
    model.wv.save(f'{output}_{size}.bin')


if __name__ == "__main__":
    # change path and parameters if necessary
    tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file("tokenizer/tokenizer_en.save")

    break_file_into_subwords("data/no_punctuation/unaligned.en", "data/no_punctuation/unaligned_subwords.en", tokenizer)

    word2vec("data/no_punctuation/unaligned_subwords.en",
             "src/embedding_models/word2vec/english_w2v_subwords",
             size=256,
             workers=8,
             min_count=1)
