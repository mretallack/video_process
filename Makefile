
PYTHON=python3.11

run: venv/bin/activate
	. venv/bin/activate && ${PYTHON} video_analyzer.py


venv/bin/activate: requirements.txt
	${PYTHON} -m venv venv
	. venv/bin/activate && ${PYTHON} -m pip install -r requirements.txt
	