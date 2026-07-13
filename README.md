# Whisper STT Lab

Run the local microphone-based benchmark UI:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:5000`, allow microphone access, record speech, then select **Transcribe**.

The first transcription downloads/loads the `small` model and reports its load time. Later recordings reuse it. Add a manually corrected reference transcript before transcribing to view normalized Word Error Rate (WER) and Character Error Rate (CER).
