# Project purpose

This is a Python application for transcribing audio with Whisper.

# Agent responsibilities

- Diagnose and fix bugs.
- Implement requested features.
- Keep changes small and readable.
- Update README.md when behavior changes.
- Never delete user files or recordings.

# Project commands

Install dependencies:

    pip install -r requirements.txt

Run the application:

    python app.py

Run tests:

    pytest

# Verification

- Check modified Python files for syntax errors.
- Run relevant tests.
- Explain any verification that could not be completed.

# Coding conventions

- Use type hints for new functions.
- Keep transcription logic in transcribe.py.
- Keep application/UI logic in app.py.
- Do not commit secrets, API keys, recordings, or generated transcripts.