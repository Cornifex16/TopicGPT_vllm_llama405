unzip "data/input/wiki_test.jsonl.zip"

pyenv install 3.10.4

pyenv local 3.10.4

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

python gen_1.py > log_gen.txt

python ref.py > log_ref.txt

python assignment_chunks.py > log_asig_wiki.txt

python assignment_chunks_bills.py > log_asig_bills.txt