pyenv install 3.10.4

pyenv local 3.10.4

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

python prueba.py > log_test.txt
