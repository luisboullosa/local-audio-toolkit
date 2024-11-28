# Audio Transcription and Summarization Tool

A command-line tool for transcribing and summarizing audio from various sources including YouTube videos, local MP3 files, and microphone recordings. Uses MLX Whisper for transcription and local LLM for summarization.

## Usage

> [!IMPORTANT]
> Make sure you have LM Studio running locally on port 1234 for the summarization feature to work.

The tool provides these commands:

### List Devices
```bash
uv run python src/main.py list-devices
```

### Transcribe
Record from microphone:
```bash
uv run python src/main.py transcribe record
```

From YouTube:
```bash
uv run python src/main.py transcribe https://youtube.com/watch?v=...
```

From local MP3:
```bash
uv run python src/main.py transcribe path/to/audio.mp3
```

### Summarise
Record from microphone:
```bash
uv run python src/main.py summarise record
```
From YouTube:
```
uv run python src/main.py summarise https://youtube.com/watch?v=...
```

From local MP3:
```
uv run python src/main.py summarise path/to/audio.mp3
```

With custom profile:
```
uv run python src/main.py summarise record --profile "software engineer writing PR description"
```

## Building

1. Clone the repository
2. Install dependencies using `uv`:
```bash
uv sync --dev --all-extras
```

## Testing

Tests can be run using pytest:
```bash
uv run pytest --cov=src tests/
```
