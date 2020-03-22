"""
Generate predictions from model and evaluate bleu score
"""
import argparse
import os
import subprocess
import tempfile

import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

from src.models.attention_gru import Decoder, Encoder
from src.utils.data_utils import preprocess_sentence

# The following config setting is necessary to work on my local RTX2070 GPU
# Comment if you suspect it's causing trouble
tf_config = ConfigProto()
tf_config.gpu_options.allow_growth = True
session = InteractiveSession(config=tf_config)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def generate_predictions(input_file_path: str, pred_file_path: str):
    """Generates predictions for the machine translation task (EN->FR).
    You are allowed to modify this function as needed, but one again, you cannot
    modify any other part of this file. We will be importing only this function
    in our final evaluation script. Since you will most definitely need to import
    modules for your code, you must import these inside the function itself.
    Args:
        input_file_path: the file path that contains the input data.
        pred_file_path: the file path where to store the predictions.
    Returns: None
    """
    DEBUG = True  # Write predictions to debug_predictions if True
    BATCH_SIZE = 64
    embedding_dim = 200
    units = 512

    # Importing language model
    inp_lang_model = KeyedVectors.load("src/embedding_models/word2vec/english_w2v_200.bin", mmap='r')
    # Add unknown token to input vocabulary with mean vector as value
    inp_mean_vector = np.mean(inp_lang_model[list(inp_lang_model.vocab.keys())], axis=0)
    inp_lang_model.add(["<unk>"], [inp_mean_vector])
    vocab_inp_size = len(inp_lang_model.vocab.keys())

    targ_lang_model = KeyedVectors.load("src/embedding_models/word2vec/french_w2v_200.bin", mmap='r')
    # Add unknown token to target vocabulary with mean vector as value
    targ_mean_vector = np.mean(targ_lang_model[list(targ_lang_model.vocab.keys())], axis=0)
    targ_lang_model.add(["<unk>"], [targ_mean_vector])
    vocab_tar_size = len(targ_lang_model.vocab.keys())

    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE, inp_lang_model)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, targ_lang_model)
    optimizer = tf.keras.optimizers.Adam()
    checkpoint_dir = './training_checkpoints'
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    def evaluate(sentence: str, max_length_targ: int) -> str:
        """
        Translate a sentence from english to french
        :param sentence: input sentence
        :param max_length_targ: maximum length of translation output
        :return: the sentence translated in french
        """
        sentence = preprocess_sentence(sentence)
        tokens = []
        for word in sentence.split():
            # transform word to its index (with +1 offset) in the language model
            if word in inp_lang_model.vocab.keys():
                tokens.append(inp_lang_model.vocab[word].index + 1)
            else:  # treat word as unknown if not in language model
                tokens.append(inp_lang_model.wv.vocab["<unk>"].index + 1)
        tensor = tf.convert_to_tensor(tokens)
        tensor = tf.reshape(tensor, (1, -1))
        result = ''

        hidden = [tf.zeros((1, units))]
        enc_out, enc_hidden = encoder(tensor, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang_model.vocab['<start>'].index + 1], 0)

        for _ in range(max_length_targ):
            predictions, dec_hidden, _ = decoder(dec_input,
                                                 dec_hidden,
                                                 enc_out)

            #  Prevent outputting <unk> (Add +1 to match the decoder embedding indexing
            unk_idx = targ_lang_model.vocab["<unk>"].index + 1
            predictions = predictions.numpy()
            predicted_id = np.argmax(predictions[0])
            predictions[0, unk_idx] = -1e9
            predicted_id_no_unk = np.argmax(predictions[0])
            predicted_word_no_unk = targ_lang_model.index2word[predicted_id_no_unk-1]

            if targ_lang_model.index2word[predicted_id-1] == '<end>':
                result += "\n"
                return result
            result += predicted_word_no_unk + ' '

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)
        result += '\n'  # add end token if max length reached
        return result

    results = []
    num_lines = sum(1 for line in open(input_file_path))
    with open(input_file_path, "r") as input_file:
        count = 0
        input_sentence = input_file.readline()
        while input_sentence:
            if count % 100 == 0:
                print(f"{count}/{num_lines}")
            # Predict maximum length of 1.25 time the input length
            # TODO See if there's a better heuristic
            max_length_targ = int(len(input_sentence.split()) * 1.25)
            result = evaluate(input_sentence, max_length_targ)
            results.append(result)
            count += 1
            input_sentence = input_file.readline()

    with open(pred_file_path, "w") as pred_file:
        for result in results:
            pred_file.write(result)
    if DEBUG:
        with open("debug_predictions", "w") as pred_file:
            for result in results:
                pred_file.write(result)

    # MODIFY ABOVE #####


def compute_bleu(pred_file_path: str, target_file_path: str, print_all_scores: bool):
    """
    Args:
        pred_file_path: the file path that contains the predictions.
        target_file_path: the file path that contains the targets (also called references).
        print_all_scores: if True, will print one score per example.
    Returns: None
    """
    out = subprocess.run(["sacrebleu", "--input", pred_file_path, target_file_path, '--tokenize',
                          'none', '--sentence-level', '--score-only'],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = out.stdout.split('\n')
    if print_all_scores:
        print('\n'.join(lines[:-1]))
    else:
        scores = [float(x) for x in lines[:-1]]
        print('final avg bleu score: {:.2f}'.format(sum(scores) / len(scores)))


def main():
    parser = argparse.ArgumentParser('script for evaluating a model.')
    parser.add_argument('--target-file-path', help='path to target (reference) file', required=True)
    parser.add_argument('--input-file-path', help='path to input file', required=True)
    parser.add_argument('--print-all-scores', help='will print one score per sentence',
                        action='store_true')
    parser.add_argument('--do-not-run-model',
                        help='will use --input-file-path as predictions, instead of running the '
                             'model on it',
                        action='store_true')

    args = parser.parse_args()

    if args.do_not_run_model:
        compute_bleu(args.input_file_path, args.target_file_path, args.print_all_scores)
    else:
        _, pred_file_path = tempfile.mkstemp()
        generate_predictions(args.input_file_path, pred_file_path)
        compute_bleu(pred_file_path, args.target_file_path, args.print_all_scores)


if __name__ == '__main__':
    main()
