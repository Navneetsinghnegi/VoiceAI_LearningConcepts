import torch
import whisper
import librosa
import numpy as np
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

whisper_model = whisper.load_model("base")
vad_model = load_silero_vad()

audio_path  = "../422-122949-0000.flac"
audio, sr = librosa.load(audio_path,sr=16000)

audio_tensor = torch.from_numpy(audio)

speech_timestamps = get_speech_timestamps(audio_tensor,vad_model,sampling_rate=16000)

print(f"Detected {len(speech_timestamps)} speech segments\n")

full_transcript = ""

for idx, segment in enumerate(speech_timestamps):
    start  = segment["start"]
    end = segment["end"]

    speech_chunk = audio[start:end]
    result = whisper_model.transcribe(speech_chunk)
    text = "".join(result["text"]).strip()

    print(f"[Segment {idx+1}] {text}")

    full_transcript += " " + text

print("\nFinal Transcript:")
print(full_transcript.strip())