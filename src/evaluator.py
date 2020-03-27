"""
Generate predictions from model and evaluate BLEU score
"""
import argparse
import json
import os
import subprocess
import tempfile

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

from src.models.Transformer import Transformer
from src.utils.transformer_utils import translate

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
    config_path = "config_files/transformer_eval_back_cfg.json"
    assert os.path.isfile(config_path), f"invalid config file: {config_path}"
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    debug = config["debug"]  # Write predictions to debug_predictions if True
    # Set hyperparameters
    num_layers = config["num_layers"]
    d_model = config["d_model"]
    dff = config["dff"]
    num_heads = config["num_heads"]
    tokenizer_source_path = config["tokenizer_source_path"]
    tokenizer_target_path = config["tokenizer_target_path"]
    dropout_rate = config["dropout_rate"]
    checkpoint_path_best = config["checkpoint_path_best"]

    tokenizer_source = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_source_path)
    tokenizer_target = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_target_path)

    source_vocab_size = tokenizer_source.vocab_size + 2
    target_vocab_size = tokenizer_target.vocab_size + 2

    transformer = Transformer(num_layers, d_model, num_heads, dff,
                              source_vocab_size, target_vocab_size,
                              pe_input=source_vocab_size,
                              pe_target=target_vocab_size,
                              rate=dropout_rate)

    ckpt = tf.train.Checkpoint(transformer=transformer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path_best, max_to_keep=1)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored from ', checkpoint_path_best)

    results = []
    num_lines = sum(1 for _ in open(input_file_path))
    # TODO check how to make this faster
    with open(input_file_path, "r") as input_file:
        count = 0
        input_sentence = input_file.readline().strip()
        while input_sentence:
            if count % 100 == 0:
                print(f"{count}/{num_lines}")
            # Predict maximum length of 1.5 time the input length
            # TODO See if there's a better heuristic
            max_length_pred = int(len(tokenizer_source.encode(input_sentence)) * 1.5)
            result = translate(input_sentence, tokenizer_source, tokenizer_target, max_length_pred, transformer)
            results.append(result)
            count += 1
            input_sentence = input_file.readline()
    with open(pred_file_path, "w") as pred_file:
        for result in results:
            pred_file.write(result + '\n')
    if debug:
        with open("debug_predictions", "w") as debug_file:
            for result in results:
                debug_file.write(result + '\n')
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
