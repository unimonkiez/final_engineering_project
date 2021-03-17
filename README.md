# final_engineering_project

## Setup

- Install [VSCode](https://code.visualstudio.com/).
- Install Python [3.8+].
- [Install Poetry](https://python-poetry.org/docs/#installation).

## Installation

- open terminal in dir.
- `poetry install`.
- Run `poetry show -v` in root and paste python path in `./.vscode/settings.json` under `"python.pythonPath"`.
  - In linux, simply run this instead -
    ```bash
    cat .vscode/settings.json | sed "s|\"python.pythonPath\": \".*\"|\"python.pythonPath\": \"`poetry show -v | grep -r "virtualenv"  | sed 's/.*: //1'`/bin/python\"|1" > ./.vscode/settings.json
    ```

## Usage

- Everything can be done from VSCode (Such as debug, test and etc).
- Couple of CLI tools:
  - `poetry run start` - start app.
  - `poetry run black .` - Format code.
  - `poetry run flake8 .` - Lint code.
  - `poetry run pytest ./tests` - Run tests.
