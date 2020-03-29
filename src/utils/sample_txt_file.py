import random
import argparse


def sample_lines_from_txt_file(input_file_path: str, out_file_path: str, num_lines: int):
    # TODO: Maybe use a better heuristic than random sampling ?
    with open(input_file_path, "r", encoding="utf-8") as f:
        lines = random.sample(f.readlines(), num_lines)

    with open(out_file_path, "w", encoding="utf-8") as f_out:
        for line in lines:
            f_out.write(line)


def main():
    parser = argparse.ArgumentParser('Script to sample randomly a fixed number of lines from text file')
    parser.add_argument('-i', '--input_file_path', type=str, help='path to input file', required=True)
    parser.add_argument('-o', '--out_file_path', type=str,
                        help='path to write random lines: defaults to name of input file path + number of samples')
    parser.add_argument('-n', '--num_lines', type=int, help='number of lines to sample', required=True)
    args = parser.parse_args()

    if not args.out_file_path:
        out_file_path = f"{args.input_file_path}_{args.num_lines}"
        print(f"out_file_path not provided -> Defaulting to {out_file_path}")
    else:
        out_file_path = args.out_file_path

    sample_lines_from_txt_file(
        args.input_file_path,
        out_file_path,
        args.num_lines
    )


if __name__ == '__main__':
    main()
