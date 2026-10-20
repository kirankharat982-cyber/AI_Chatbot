# Offline GenAI Chatbot

A simple command-line AI chatbot built with Python and Hugging Face Transformers. It uses the `HuggingFaceTB/SmolLM2-360M-Instruct` language model and is configured to run on the CPU.

## Features

- Answers questions in simple English.
- Maintains a short conversation history.
- Supports clearing conversation history with `/reset`.
- Supports exiting the chatbot with `/exit`.
- Loads the model from Hugging Face on the first run and can use the local cache on later runs.

## Project Files

```text
AI_Project/
├── AI_Project.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or newer is recommended.
- An internet connection for the first model download.
- Enough RAM and disk space for PyTorch and the language model.

## Setup and Run in VS Code

### 1. Open the project

Open the folder containing `AI_Project.py` in Visual Studio Code.

### 2. (Optional) Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

If PowerShell blocks activation, you can use Command Prompt or run the install command with the environment's Python directly.

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the chatbot

```bash
python AI_Project.py
```

On the first run, the tokenizer and model are downloaded from Hugging Face. This requires an internet connection and may take several minutes. Later runs can usually use the cached files.

## How to Use

When the chatbot starts, type a question after `You:`.

- Type `/reset` to clear the conversation history.
- Type `/exit` to quit the chatbot.

## Technologies Used

- Python
- PyTorch (`torch`)
- Hugging Face Transformers
- SmolLM2-360M-Instruct

## Troubleshooting

### `python` is not recognized

Install Python from [python.org](https://www.python.org/downloads/), then restart VS Code and check:

```bash
python --version
```

### Missing package errors

Install the dependencies again:

```bash
python -m pip install -r requirements.txt
```

### Model download fails

Check your internet connection and retry. The first run needs access to Hugging Face to download the model.

### The chatbot is slow

The project runs the model on CPU, so response generation may be slower on some computers.

## Notes

- The chatbot generates responses with a language model; answers may be inaccurate, so verify important information.
- This is a terminal-based chatbot, not a web application.
