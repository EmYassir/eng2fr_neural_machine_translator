import argparse
import tempfile
import os
import tensorflow as tf

from src.evaluator import generate_predictions, compute_bleu
from src.utils.data_utils import project_root


def eval_by_length(
        input_file_path: str,
        target_file_path: str,
        saved_path: str,
        config_file: str,
        print_all_scores: bool
):
    with open(input_file_path, "r", encoding="utf-8") as f_in:
        input_lines = f_in.readlines()
    with open(target_file_path, "r", encoding="utf-8") as f_in:
        target_lines = f_in.readlines()

    lengths = [0, 10, 20, 30, 40, 10000]
    for i in range(len(lengths) - 1):
        index_lines_eval = [
            input_lines.index(line) for line in input_lines if lengths[i] < len(line.split(" ")) < lengths[i + 1]
        ]

        tf.print(f"{len(index_lines_eval)} sentence(s) with n_words between {lengths[i]} and {lengths[i + 1]}")
        if len(index_lines_eval) == 0:
            continue

        tmp_input = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
        tmp_preds = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
        tmp_target = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
        tmp_preds.close()

        tf.print(f"Writing input sentences to {tmp_input.name}")
        for index in index_lines_eval:
            tmp_input.write(input_lines[index])
            tmp_target.write(target_lines[index])
        tmp_input.close()
        tmp_target.close()

        generate_predictions(
            tmp_input.name,
            tmp_preds.name,
            saved_path,
            config_file
        )
        compute_bleu(tmp_preds.name, tmp_target.name, print_all_scores)

        for tmp in [tmp_input, tmp_preds, tmp_target]:
            tmp.close()
            os.remove(tmp.name)


def main():
    parser = argparse.ArgumentParser('Script for evaluating a model according to sequence lengths')
    parser.add_argument('--input-file-path', '-i', type=str, help='path to input file', required=True)
    parser.add_argument('--target-file-path', '-t', type=str, help='path to target file', required=True)
    parser.add_argument('--config_file', '-c', type=str,
                        help='path to config file',
                        default=os.path.join("config_files", "transformer_eval_cfg.json"))
    parser.add_argument('--saved_path', '-s', type=str,
                        help='path to saved models/tokenizers', default=project_root())
    parser.add_argument('--print-all-scores', help='will print one score per sentence',
                        action='store_true')
    args = parser.parse_args()
    eval_by_length(
        args.input_file_path,
        args.target_file_path,
        args.saved_path,
        args.config_file,
        args.print_all_scores
    )


if __name__ == '__main__':
    main()
