import unittest
import os
import json
from shutil import rmtree

from src.train_transformer import train_transformer
from src.evaluator import generate_predictions, compute_bleu
from src.utils.data_utils import project_root


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.input_file = os.path.join(project_root(), "tests", "train_test.lang1")
        self.target_file = os.path.join(project_root(), "tests", "train_test.lang2")
        self.pred_file = os.path.join(project_root(), "tests", "train_test.pred2")
        self.config_train_path = os.path.join(project_root(), "config_files", "debug", "transformer_cfg.json")
        self.config_eval_path = os.path.join(project_root(), "config_files", "debug", "transformer_eval_cfg.json")

    def test_train_and_eval_transformer(self):
        """ Test training and evaluating tiny transformer"""
        train_transformer(
            config_path=self.config_train_path,
            data_path=project_root(),
            save_path=project_root(),
            restore_checkpoint=False,
            print_all_scores=True
        )

        # Checkpoint of trained model should be saved by now
        # -> Generate predictions and compute bleu score
        for n_lines in [1, None]:
            # Try that n_lines argument works
            generate_predictions(
                self.input_file, self.pred_file, project_root(),
                self.config_eval_path, max_lines_process=n_lines
            )

        compute_bleu(self.pred_file, self.target_file, print_all_scores=True)

        with open(self.config_train_path, "r") as f_in:
            config = json.load(f_in)

        # Delete temporary files when done.
        rmtree(os.path.join(project_root(), config["checkpoint_path"]))
        rmtree(os.path.join(project_root(), config["checkpoint_path_best"]), ignore_errors=True)
        os.remove(self.pred_file)


if __name__ == '__main__':
    unittest.main()
