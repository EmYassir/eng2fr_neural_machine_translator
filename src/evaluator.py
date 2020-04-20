"""
Generate predictions from model and evaluate BLEU score
"""
import argparse
import json
import os
import subprocess
import tempfile
import time
from typing import Optional

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

from src.utils.transformer_utils import load_transformer, translate_file
from src.utils.data_utils import project_root

# The following config setting is necessary to work on my local RTX2070 GPU
# Comment if you suspect it's causing trouble
tf_config = ConfigProto()
tf_config.gpu_options.allow_growth = True
session = InteractiveSession(config=tf_config)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def generate_predictions(
        input_file_path: str,
        pred_file_path: str,
        saved_path: str,
        config_file: str,
        max_lines_process: Optional[int] = None
):
    """Generates predictions for the machine translation task (EN->FR).
    You are allowed to modify this function as needed, but one again, you cannot
    modify any other part of this file. We will be importing only this function
    in our final evaluation script. Since you will most definitely need to import
    modules for your code, you must import these inside the function itself.
    Args:
        input_file_path: the file path that contains the input data.
        pred_file_path: the file path where to store the predictions.
        saved_path: path to directory where models/tokenizers are
        config_file: name of config file
        max_lines_process: maximum number of lines to translate (all of them if not specified)
    Returns: None
    """
    start = time.time()
    tf.print(f"Using config_file={config_file}")
    if os.path.exists(config_file):
        config_path = config_file
    else:
        # if not a path, check project folder
        config_path = os.path.join(project_root(), config_file)

    assert os.path.isfile(config_path), f"invalid config file: {config_path}"

    with open(config_path, "r") as config_f:
        config = json.load(config_f)

    if "debug" not in config:
        tf.print(f"Warning: debug not in config -> Defaulting to False")
        debug = False
    else:
        debug = config["debug"]  # Write predictions to debug_predictions if True
    if "beam_size" not in config:
        tf.print(f"Warning: beam_size not in config -> Defaulting to None")
        beam_size = None
    else:
        beam_size = config["beam_size"]

    if "alpha" not in config:
        tf.print(f"Warning: alpha not in config -> Defaulting to None")
        alpha = None
    else:
        alpha = config["alpha"]
    tokenizer_source_path = os.path.join(saved_path, config["tokenizer_source_path"])
    tokenizer_target_path = os.path.join(saved_path, config["tokenizer_target_path"])
    checkpoint_path_best = os.path.join(saved_path, config["checkpoint_path_best"])
    if "translation_batch_size" not in config:
        tf.print(f"Warning: translation_batch_size not in config -> Defaulting to 32")
        translation_batch_size = 32
    else:
        translation_batch_size = config["translation_batch_size"]
    tokenizer_source = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_source_path)
    tokenizer_target = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_target_path)

    transformer = load_transformer(config, tokenizer_source, tokenizer_target)

    ckpt = tf.train.Checkpoint(transformer=transformer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path_best, max_to_keep=1)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        tf.print('Latest checkpoint restored from ', checkpoint_path_best)

    num_lines = sum(1 for _ in open(input_file_path))
    if max_lines_process is not None:
        num_lines = min(num_lines, max_lines_process)

    tf.print(f"Translating a total of {num_lines} sentences")

    # Get translations in order of sentences length and dict to re-order them later
    results, sorted_keys = translate_file(transformer, tokenizer_source, tokenizer_target,
                                          input_file_path, batch_size=translation_batch_size,
                                          max_lines_process=max_lines_process,
                                          beam_size=beam_size,
                                          alpha=alpha)

    # Write predictions in the right order
    if debug:
        with open("debug_predictions", "w", encoding="utf-8") as f_out:
            with open(input_file_path, "r") as f_in:
                for index in range(len(results)):
                    f_out.write(f"Input:{f_in.readline()}\n")
                    f_out.write(f"Output: {results[sorted_keys[index]]}\n\n")
                    f_out.write("-----------------------------------------------------------\n\n")
    tf.print(f"Writing predictions to path = {pred_file_path}")
    with open(pred_file_path, "w", encoding="utf-8") as f_out:
        for index in range(len(sorted_keys)):
            f_out.write(f"{results[sorted_keys[index]]}\n")
    tf.print(f"Time for prediction: {time.time() - start} seconds")


def compute_bleu(pred_file_path: str, target_file_path: str, print_all_scores: bool):
    """
    Args:
        pred_file_path: the file path that contains the predictions.
        target_file_path: the file path that contains the targets (also called references).
        print_all_scores: if True, will print one score per example.
    Returns: None
    """
    tf.print("Starting to compute bleu score...")
    out = subprocess.run(["sacrebleu", "--input", pred_file_path, target_file_path, '--tokenize',
                          'none', '--sentence-level', '--score-only'],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = out.stdout.split('\n')
    if print_all_scores:
        tf.print('\n'.join(lines[:-1]))
    else:
        scores = [float(x) for x in lines[:-1]]
        tf.print('final avg bleu score: {:.2f}'.format(sum(scores) / len(scores)))


def main():
    parser = argparse.ArgumentParser('script for evaluating a model.')
    parser.add_argument('--target-file-path', type=str, help='path to target (reference) file', required=True)
    parser.add_argument('--input-file-path', type=str, help='path to input file', required=True)
    parser.add_argument('--config_file', type=str,
                        help='path to config file',
                        default=os.path.join("config_files", "transformer_eval_cfg.json"))
    parser.add_argument('--saved_path', type=str, help='path to saved models/tokenizers', default=project_root())
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
        generate_predictions(args.input_file_path, pred_file_path, args.saved_path, args.config_file)
        compute_bleu(pred_file_path, args.target_file_path, args.print_all_scores)


if __name__ == '__main__':
    main()
