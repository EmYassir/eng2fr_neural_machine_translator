import unittest
import os

from src.eval_by_length import eval_by_length
from src.utils.data_utils import project_root


class MyTestCase(unittest.TestCase):

    @staticmethod
    def test_eval_by_length():
        """ Just make sure no error is thrown """
        eval_by_length(
            os.path.join(project_root(), "data", "debug.lang1"),
            os.path.join(project_root(), "data", "debug.lang2"),
            project_root(),
            os.path.join(project_root(), "config_files", "debug", "transformer_eval_cfg.json"),
            print_all_scores=True
        )


if __name__ == '__main__':
    unittest.main()
