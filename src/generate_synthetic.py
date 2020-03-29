import argparse

from src.evaluator import generate_predictions
from src.utils.data_utils import project_root


def main():
    parser = argparse.ArgumentParser('Script to generate synthetic data')
    parser.add_argument('-p', '--pred_file_path', type=str, help='path to target (reference) file', required=True)
    parser.add_argument('-i', '--input_file_path', type=str, help='path to input file', required=True)
    parser.add_argument('-c', '--config_file', type=str,
                        help='name of config file in directory config_files/', required=True)
    parser.add_argument('-s', '--saved_path', type=str, help='path to saved models/tokenizers', default=project_root())
    parser.add_argument('-n', '--num_lines', type=int, help='number of lines to translate', default=5)
    args = parser.parse_args()

    generate_predictions(
        args.input_file_path,
        args.pred_file_path,
        args.saved_path,
        args.config_file,
        args.num_lines
    )


if __name__ == '__main__':
    main()
