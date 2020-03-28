# Requirements
Any package needed to run the code should be added to requirements.txt

Packages needed only for developpement such as flake8 and pytest should be added to requirements_dev.txt

# PEP8
To ensure that all code is PEP8 compliant, flake8 will run on all code pushed to this repository. 
You can check locally that all code is PEP8 compliant by running 
```bash
flake8
```
from the root folder.

# Pytest
To ensure that no change break previous working code, pytest will run every time code is pushed to github.
Feel free to add tests to ensure that every functions have the expected behaviour at all time.
You can check locally that your coding is passing tests by running 
```bash
python -m pytest
```

# PR
All change to the master branch should be made through a pull request. 
The approval of one collaborator is mandatory for a PR to be merged to the master branch.
The PR should pass the flake8 and pytest auto-runs without failure before being merged to master. 

# Steps to reproduce training with attention GRU (see jerome_dev branch for evaluator function)
* Tokenize the unaligned data in both language and remove punctuation.
* Add <start> and <end> tokens for each line
* Train a language model (ex: Word2Vec with the src.word_embeddings_models/word2vec.py script) on both languages
* Tokenize the .lang2 files and remove punctuation to have a vocabulary similar to language model
* Run the train.py script with training.lang1 and the cleaned version of training.lang2
* You can then evaluate the model by feeding it the validation.lang1 file and compute the blue score on the validation.lang2 or its cleaned version.
