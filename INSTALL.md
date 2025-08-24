# Installation

## Prerequisites

### Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Packages

```bash
brew install ffmpeg python3.12
```

### Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Python

### Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### VSCode

For ease of use the environment can optionally be selected for all new terminal windows as default:
* Press Shift+Command+P for commands
* Then search for "Python: Select Interpreter"
* And choose recently created, mostly named "Python ... (.venv) ./venv/bin/python Recommended"

### Packages

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run src/app.py
```
