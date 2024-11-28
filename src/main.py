import os
import struct
import time
import wave
from pathlib import Path
from typing import Optional

import mlx_whisper
import typer
from ffmpeg import FFmpeg
from openai import OpenAI
from pvrecorder import PvRecorder
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from yt_dlp import YoutubeDL

app = typer.Typer(help="Audio recording and processing tool")


def is_youtube_url(url: str) -> bool:
    """Check if the input is a YouTube URL."""
    return url.startswith(
        (
            "https://www.youtube.com/",
            "https://youtube.com/",
            "https://youtu.be/",
        )
    )


def download_from_youtube(url: str, audios_path: Path) -> Path:
    """Download audio from a YouTube video."""
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": f"{audios_path}/%(id)s.%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "logger": None,
    }

    spinner = Spinner("dots", text="Downloading from YouTube...")
    with Live(spinner, refresh_per_second=10):
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            id = info["id"]
            mp3_path = audios_path / f"{id}.mp3"
            spinner.text = "Download completed"
            return mp3_path


def record_audio(audios_path: Optional[Path], device_index: int = -1) -> Path:
    recorder = PvRecorder(frame_length=512, device_index=device_index)

    recorder.start()

    timestamp = int(time.time())
    temp_wav = audios_path / f"temp_{timestamp}.wav"
    final_mp3 = audios_path / f"recording_{timestamp}.mp3"
    wavfile = None

    try:
        if audios_path is not None:
            wavfile = wave.open(str(temp_wav), "w")
            wavfile.setparams(
                (
                    1,
                    2,
                    recorder.sample_rate,
                    recorder.frame_length,
                    "NONE",
                    "NONE",
                )
            )

        spinner = Spinner("dots", text="Recording... Press Ctrl+C to stop")
        with Live(spinner, refresh_per_second=10):
            while True:
                frame = recorder.read()
                if wavfile is not None:
                    wavfile.writeframes(struct.pack("h" * len(frame), *frame))
    except KeyboardInterrupt:
        typer.echo("Stopping recording")

    recorder.delete()
    if wavfile is not None:
        wavfile.close()

    ffmpeg = (
        FFmpeg()
        .option("y")
        .input(str(temp_wav))
        .output(
            str(final_mp3),
            {"acodec": "libmp3lame"},
        )
    )

    typer.echo(f"Converting {temp_wav.absolute()} to {final_mp3.absolute()}...")
    ffmpeg.execute()

    os.remove(temp_wav)

    return final_mp3


def transcribe_audio(speech_file: Path) -> str:
    spinner = Spinner("dots", text="Transcribing audio...")
    with Live(spinner, refresh_per_second=10):
        result = mlx_whisper.transcribe(
            str(speech_file),
            # FIX: `ValueError: [load_npz] Input must be a zip file or a file-like object that can be opened with zipfile.ZipFile`
            # path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
            path_or_hf_repo="mlx-community/whisper-turbo",
        )

    # Show preview of first 100 characters
    console = Console()
    preview = (
        result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"]
    )
    console.print(f"Transcription preview: {preview}")

    return result["text"]


def summarise_audio(text: str, profile: Optional[str] = None):
    # Initialize OpenAI client that points to the local LM Studio server
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    # Calculate number of words
    word_count = len(text.split())

    # Determine number of summary points based on length
    if word_count < 500:
        points_range = "2-3"
    elif word_count < 2000:
        points_range = "3-5"
    else:
        points_range = "5-7"

    # Build the prompt based on profile
    base_prompt = f"Summarise this text in {points_range} key points, with a brief introduction. For longer summaries, ensure the points cover different aspects of the content."

    if profile:
        system_message = f"You are an AI assistant helping a {profile}."
        user_prompt = f"As a {profile}, {base_prompt}"
    else:
        system_message = "You are a helpful AI assistant."
        user_prompt = base_prompt

    # Define the conversation with the AI
    messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": f"{user_prompt}\n\nText to summarize:\n{text}",
        },
    ]

    spinner = Spinner("dots", text="Generating summary...")
    with Live(spinner, refresh_per_second=10):
        # Get response from AI
        response = client.chat.completions.create(
            model="llama-3.2-3b-instruct",
            messages=messages,
        )

    # Parse and display the results
    results = response.choices[0].message.content
    console = Console()
    console.print(f"\nSummary ({word_count} words in original text):")
    console.print(results)


def validate_local_file(file_path: Path) -> None:
    """Validate that the file exists and is an MP3."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.suffix.lower() != ".mp3":
        raise ValueError("File must be an MP3")
    return file_path


@app.command()
def summarise(
    source: Optional[str] = typer.Argument(
        None,
        help="YouTube URL, path to local MP3 file, or 'record' for microphone recording",
    ),
    audios_path: Path = typer.Option(
        Path("audios"),
        "--audios",
        "-a",
        help="Directory to save the audio file",
    ),
    transcripts_path: Path = typer.Option(
        Path("transcripts"),
        "--transcripts",
        "-t",
        help="Directory to save the transcript file",
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        "-p",
        help="Specify a profile (e.g., 'software engineer writing PR description')",
    ),
    device_index: int = typer.Option(
        1,
        "--device",
        "-d",
        help="Audio input device index",
    ),
):
    """Summarise audio from a source."""
    audios_path.mkdir(exist_ok=True)
    transcripts_path.mkdir(exist_ok=True)

    audio_file = None
    if source.lower() == "record":
        audio_file = record_audio(
            audios_path=audios_path,
            device_index=device_index,
        )
        typer.echo(f"Audio recorded and saved to {audio_file}")
    elif is_youtube_url(source):
        audio_file = download_from_youtube(source, audios_path)
        typer.echo(f"Audio downloaded from YouTube to {audio_file}")
    else:
        source_path = Path(source)
        try:
            audio_file = validate_local_file(source_path)
            typer.echo(f"Valid MP3 file: {audio_file}")
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Error: {str(e)}", err=True)
            raise typer.Exit(1)

    if audio_file:
        transcribed_text = transcribe_audio(audio_file)
        summarise_audio(transcribed_text, profile)


@app.command()
def list_devices():
    """List all available audio input devices."""
    devices = PvRecorder.get_available_devices()
    for i, device in enumerate(devices):
        typer.echo(f"{i}: {device}")


if __name__ == "__main__":
    app()
