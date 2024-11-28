from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from src.main import (
    is_youtube_url,
    record_audio,
    summarise_audio,
    transcribe_audio,
    validate_local_file,
)

runner = CliRunner()


def test_is_youtube_url():
    assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True
    assert is_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ") is True
    assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ") is True
    assert is_youtube_url("https://example.com") is False
    assert is_youtube_url("not_a_url") is False


def test_validate_local_file(tmp_path):
    # Create a temporary MP3 file
    mp3_file = tmp_path / "test.mp3"
    mp3_file.touch()

    # Create a non-MP3 file
    txt_file = tmp_path / "test.txt"
    txt_file.touch()

    # Test valid MP3 file
    assert validate_local_file(mp3_file) == mp3_file

    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        validate_local_file(tmp_path / "nonexistent.mp3")

    # Test non-MP3 file
    with pytest.raises(ValueError):
        validate_local_file(txt_file)


def test_record_audio(tmp_path):
    timestamp = 1234567890
    audios_path = tmp_path / "audios"
    audios_path.mkdir()

    mock_recorder = Mock()
    mock_recorder.sample_rate = 16000
    mock_recorder.frame_length = 128

    # Make read() raise KeyboardInterrupt after 3 frames
    read_count = 0

    def mock_read():
        nonlocal read_count
        if read_count >= 3:
            raise KeyboardInterrupt
        read_count += 1
        return [0] * 128

    mock_recorder.read = mock_read

    with patch("src.main.PvRecorder", return_value=mock_recorder):
        with patch("src.main.FFmpeg") as mock_ffmpeg:
            mock_ffmpeg.return_value.option.return_value.input.return_value.output.return_value.execute.return_value = None

            result = record_audio(audios_path, timestamp, device_index=1)

            assert result == audios_path / f"recording_{timestamp}.mp3"
            assert mock_recorder.start.called
            assert mock_recorder.delete.called


def test_transcribe_audio(tmp_path):
    test_file = tmp_path / "test.mp3"
    test_file.touch()

    expected_text = "This is a test transcription"
    mock_result = {"text": expected_text}

    with patch("src.main.mlx_whisper.transcribe", return_value=mock_result):
        result = transcribe_audio(test_file)
        assert result == expected_text


def test_summarise_audio():
    test_text = "This is a test text to summarize"
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Summary of the text"))]

    with patch("src.main.OpenAI") as mock_openai:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        summarise_audio(test_text, profile="test profile")

        mock_client.chat.completions.create.assert_called_once()


@pytest.mark.parametrize(
    "source,expected_output",
    [
        ("record", "Audio recorded"),
        ("https://youtube.com/watch?v=test", "Audio downloaded from YouTube"),
        ("test.mp3", "Valid MP3 file"),
    ],
)
def test_summarise_command(tmp_path, source, expected_output):
    with (
        patch("src.main.record_audio") as mock_record,
        patch("src.main.download_from_youtube") as mock_download,
        patch("src.main.validate_local_file") as mock_validate,
        patch("src.main.transcribe_audio") as mock_transcribe,
    ):
        # Setup mocks
        mock_record.return_value = tmp_path / "recording.mp3"
        mock_download.return_value = tmp_path / "youtube.mp3"
        mock_validate.return_value = tmp_path / "test.mp3"
        mock_transcribe.return_value = "Test transcription"

        from src.main import app

        result = runner.invoke(app, ["summarise", source])

        assert result.exit_code == 0
        assert expected_output in result.stdout


def test_list_devices():
    mock_devices = ["Device 1", "Device 2"]

    with patch("src.main.PvRecorder.get_available_devices", return_value=mock_devices):
        from src.main import app

        result = runner.invoke(app, ["list-devices"])

        assert result.exit_code == 0
        assert "Device 1" in result.stdout
        assert "Device 2" in result.stdout
