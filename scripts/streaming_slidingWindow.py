import whisper
import librosa
import numpy as np
import time
import soundfile as sf

model_size="tiny"
print("Loading model...")
model=whisper.load_model(model_size)

audio= "./audio.wav"
audio, sr = librosa.load(audio, sr=16000)

window_duration = 2
step_duration=1

window_size = int(window_duration * sr)
step_size = int(step_duration * sr)

full_transcript = ""
prev_text = ""

print("\nStarting sliding window simulation...\n")

chunk_index = 1

for start_idx in range(0, len(audio)-window_size+1, step_size):
    end_idx = start_idx + window_size
    chunk = audio[start_idx:end_idx]

    temp_file = "temp_chunk_sliding.wav"
    sf.write(temp_file,chunk,sr)

    start_time = time.time()
    result = model.transcribe(temp_file)
    end_time = time.time()

    current_text = "".join(result["text"]).strip()
    
    if current_text.startswith(prev_text):
        new_part = current_text[len(prev_text):].strip()
    else:
        new_part = current_text

    full_transcript+= " "+ new_part
    prev_text=current_text

    print(f"[Window {chunk_index}]")
    print("Current:", current_text)
    print("New Part:", new_part)
    print("Accumulated:", full_transcript.strip())
    print("Latency:", round(end_time - start_time, 2), "seconds")
    print("-" * 50)

    chunk_index += 1

print("\nFinal Transcript")
print(full_transcript.strip())