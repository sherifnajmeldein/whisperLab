from pathlib import Path
from faster_whisper import WhisperModel


AUDIO_PATH = Path("audio/test.wav")


def main() -> None:
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(f"Audio file was not found: {AUDIO_PATH}")

    print("Loading Whisper model...")

    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",
    )

    print("Transcribing audio...")

    segments, info = model.transcribe(
        str(AUDIO_PATH),
        language="ar",
        task="transcribe",
        beam_size=5,
        vad_filter=True,
    )

    print(f"Detected language: {info.language}")
    print(f"Language probability: {info.language_probability:.2f}")

    transcript_parts: list[str] = []

    for segment in segments:
        text = segment.text.strip()
        transcript_parts.append(text)

        print(
            f"[{segment.start:.2f}s -> {segment.end:.2f}s] "
            f"{text}"
        )

    full_text = " ".join(transcript_parts).strip()

    print("\nComplete transcript:")
    print(full_text)


if __name__ == "__main__":
    main()