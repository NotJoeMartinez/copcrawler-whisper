# copcrawler-whsiper
An attempt to fine tune the OpenAI Whisper model for police scanner
audio transcription on [copcrawler.com](https://copcrawler.com).

Initially using this [Fine-Tuning-Whisper-on-Custom-Dataset](https://github.com/spmallick/learnopencv/tree/master/Fine-Tuning-Whisper-on-Custom-Dataset) boilerplate.

## Installation

Requires Python 3.11.6

```bash
pyenv install 3.11.6
pyenv local 3.11.6
eval "$(pyenv init -)"
eval "$(pyenv init --path)"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```