"""train a Word2Vec model"""
from gensim.models import Word2Vec


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
            # TODO insert preprocessing here ? (stemming, lemmatization, subwords /)
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
    word2vec("data/no_punctuation/start_end_french",
             "src/embedding_models/word2vec/french_w2v",
             size=200,
             workers=8, min_count=5)
