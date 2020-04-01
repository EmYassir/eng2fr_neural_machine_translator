import unittest
import os
import glob
import json
from src.config import ConfigEvalTransformer, ConfigTrainTransformer
from src.utils.data_utils import project_root


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.path_files = os.path.join(project_root(), "config_files")
        self.all_files = [f for f in glob.glob(os.path.join(self.path_files, "**/*.json"), recursive=True)]

    def test_transformer_cfg_files_integrity(self):
        """This will throw an error if a transformer_cfg file (eval or train) has missing keys"""
        for file in self.all_files:
            print(f"File = {file}")
            if "transformer_eval" in file:
                with open(file, "r") as f_in:
                    config = json.load(f_in)
                    keys_class = ConfigEvalTransformer.__annotations__.keys()
            elif "transformer" in file:
                with open(file, "r") as f_in:
                    config = json.load(f_in)
                    keys_class = ConfigTrainTransformer.__annotations__.keys()
            else:
                continue

            self.assertListEqual(sorted(list(keys_class)), sorted(list(config.keys())))


if __name__ == '__main__':
    unittest.main()
