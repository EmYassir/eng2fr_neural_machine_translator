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
