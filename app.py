"""A small local UI for measuring faster-whisper transcription quality and speed."""

from __future__ import annotations

import os
import re
import tempfile
import threading
import time
from pathlib import Path

import av
from faster_whisper import WhisperModel
from flask import Flask, jsonify, request


# Avoid the optional Hugging Face Xet backend that can fail on some networks.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

MODEL_NAME = "small"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

_model: WhisperModel | None = None
_model_lock = threading.Lock()


def get_model() -> tuple[WhisperModel, float]:
    """Load once, then reuse the model for each recording."""
    global _model
    if _model is not None:
        return _model, 0.0

    # Flask can serve multiple uploads at once. Without this lock, concurrent
    # first requests can each allocate and load a separate Whisper model.
    with _model_lock:
        if _model is not None:
            return _model, 0.0

        start = time.perf_counter()
        _model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
        return _model, time.perf_counter() - start


def audio_duration_seconds(path: Path) -> float:
    """Read duration using PyAV, which faster-whisper already installs."""
    with av.open(str(path)) as container:
        if container.duration is None:
            return 0.0
        return float(container.duration / av.time_base)


def normalize_arabic(text: str) -> str:
    """Light normalization so Arabic spelling variants do not skew evaluation."""
    text = text.lower()
    text = re.sub(r"[\u064b-\u065f\u0670]", "", text)  # diacritics
    text = re.sub(r"[إأآ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    return " ".join(re.findall(r"[\w\u0600-\u06ff]+", text))


def edit_distance(first: list[str], second: list[str]) -> int:
    """Levenshtein distance without an extra dependency."""
    previous = list(range(len(second) + 1))
    for i, value in enumerate(first, start=1):
        current = [i]
        for j, other in enumerate(second, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (value != other),
            ))
        previous = current
    return previous[-1]


def error_rates(reference: str, transcript: str) -> dict[str, float] | None:
    if not reference.strip():
        return None
    reference = normalize_arabic(reference)
    transcript = normalize_arabic(transcript)
    words = reference.split()
    characters = list(reference.replace(" ", ""))
    wer = edit_distance(words, transcript.split()) / max(1, len(words))
    cer = edit_distance(characters, list(transcript.replace(" ", ""))) / max(1, len(characters))
    return {"wer_percent": round(wer * 100, 2), "cer_percent": round(cer * 100, 2)}


@app.get("/")
def index() -> str:
    return PAGE


@app.post("/transcribe")
def transcribe():
    recording = request.files.get("audio")
    if recording is None or not recording.filename:
        return jsonify(error="No recording was received."), 400

    suffix = Path(recording.filename).suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        recording.save(temp)
        audio_path = Path(temp.name)

    try:
        duration = audio_duration_seconds(audio_path)
        model, model_load_seconds = get_model()
        start = time.perf_counter()
        segments, info = model.transcribe(
            str(audio_path), language="ar", task="transcribe", beam_size=5, vad_filter=True
        )
        rows = [
            {"start": round(segment.start, 2), "end": round(segment.end, 2), "text": segment.text.strip()}
            for segment in segments
        ]
        transcription_seconds = time.perf_counter() - start
        transcript = " ".join(row["text"] for row in rows).strip()
        reference = request.form.get("reference", "")

        return jsonify(
            transcript=transcript,
            segments=rows,
            audio_duration_seconds=round(duration, 2),
            transcription_seconds=round(transcription_seconds, 2),
            real_time_factor=round(transcription_seconds / duration, 3) if duration else None,
            model_load_seconds=round(model_load_seconds, 2),
            detected_language=info.language,
            language_probability=round(info.language_probability * 100, 1),
            metrics=error_rates(reference, transcript),
            configuration={"model": MODEL_NAME, "device": DEVICE, "compute_type": COMPUTE_TYPE, "beam_size": 5, "vad_filter": True},
        )
    except Exception as error:
        return jsonify(error=str(error)), 500
    finally:
        audio_path.unlink(missing_ok=True)


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Whisper STT Lab</title>
<style>
  :root { color-scheme: dark; font-family: system-ui, sans-serif; background:#101827; color:#edf2f7; }
  body { max-width:960px; margin:36px auto; padding:0 20px; } h1{margin-bottom:4px} .muted{color:#9ba9c0}
  button { border:0; border-radius:8px; padding:12px 17px; margin:8px 6px 8px 0; font-weight:700; cursor:pointer; background:#38bdf8; color:#062033; } button:disabled { opacity:.45; cursor:not-allowed } #record{background:#fb7185;color:#35060d}
  textarea { box-sizing:border-box; width:100%; min-height:84px; border:1px solid #35445d; border-radius:8px; padding:10px; background:#172236; color:inherit; }
  .panel { background:#172236; padding:18px; border-radius:12px; margin-top:18px; } .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; }
  .metric { padding:13px; border-radius:9px; background:#202e45; } .metric span {display:block;color:#9ba9c0;font-size:.8rem} .metric strong {font-size:1.35rem} pre {white-space:pre-wrap; font-family:inherit; line-height:1.65}
  .hidden{display:none} #status{min-height:24px;color:#7dd3fc} audio{width:100%;margin:10px 0}
</style></head><body>
<h1>Whisper STT Lab</h1><p class="muted">Record Arabic speech, transcribe locally, and measure speed and accuracy.</p>
<div class="panel"><button id="record">● Start recording</button><button id="stop" disabled>Stop recording</button><button id="transcribe" disabled>Transcribe</button><p id="status"></p><audio id="playback" controls class="hidden"></audio></div>
<div class="panel"><label for="reference"><strong>Reference transcript (optional)</strong></label><p class="muted">Paste the correct transcript to calculate normalized WER and CER.</p><textarea id="reference" placeholder="مثال: هذا هو النص الصحيح للتسجيل"></textarea></div>
<section id="result" class="hidden"><div class="panel"><h2>Measurements</h2><div id="metrics" class="grid"></div></div><div class="panel"><h2>Transcript</h2><pre id="transcript"></pre></div><div class="panel"><h2>Segments</h2><pre id="segments"></pre></div></section>
<script>
let recorder, chunks=[], audioBlob;
const byId=id=>document.getElementById(id); const record=byId('record'), stop=byId('stop'), transcribe=byId('transcribe'), status=byId('status');
record.onclick=async()=>{ try { const stream=await navigator.mediaDevices.getUserMedia({audio:true}); chunks=[]; recorder=new MediaRecorder(stream); recorder.ondataavailable=e=>chunks.push(e.data); recorder.onstop=()=>{ audioBlob=new Blob(chunks,{type:recorder.mimeType||'audio/webm'}); byId('playback').src=URL.createObjectURL(audioBlob); byId('playback').classList.remove('hidden'); transcribe.disabled=false; stream.getTracks().forEach(track=>track.stop()); status.textContent='Recording ready. Transcribe when you are ready.'; }; recorder.start(); record.disabled=true; stop.disabled=false; status.textContent='Recording…'; } catch(e) { status.textContent='Microphone access failed: '+e.message; } };
stop.onclick=()=>{recorder.stop(); record.disabled=false; stop.disabled=true};
transcribe.onclick=async()=>{ const form=new FormData(); form.append('audio',audioBlob,'recording.webm'); form.append('reference',byId('reference').value); transcribe.disabled=true; status.textContent='Transcribing…'; const started=performance.now(); try { const response=await fetch('/transcribe',{method:'POST',body:form}); const data=await response.json(); if(!response.ok) throw new Error(data.error||'Transcription failed'); show(data); status.textContent=`Done — browser request: ${((performance.now()-started)/1000).toFixed(2)} s`; } catch(e) { status.textContent='Error: '+e.message; } finally {transcribe.disabled=false;} };
function show(d){ const values=[['Audio duration',d.audio_duration_seconds+' s'],['Transcription time',d.transcription_seconds+' s'],['Real-time factor',d.real_time_factor===null?'—':d.real_time_factor+'×'],['Language',d.detected_language+' ('+d.language_probability+'%)'],['Model',d.configuration.model+' · '+d.configuration.device+' · '+d.configuration.compute_type],['Model load',d.model_load_seconds+' s']]; if(d.metrics) values.push(['WER',d.metrics.wer_percent+'%'],['CER',d.metrics.cer_percent+'%']); byId('metrics').innerHTML=values.map(([k,v])=>`<div class="metric"><span>${k}</span><strong>${v}</strong></div>`).join(''); byId('transcript').textContent=d.transcript||'(No speech detected.)'; byId('segments').textContent=d.segments.map(s=>`[${s.start}s → ${s.end}s] ${s.text}`).join('\n'); byId('result').classList.remove('hidden'); }
</script></body></html>'''


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
