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

chunk_duration= 1
chunk_size= int(chunk_duration * sr)

full_transcript=""

print("\nStarting simulated streaming...\n")

for i in range(0,len(audio), chunk_size):
    chunk= audio[i:i + chunk_size]

    if len(chunk) ==0:
        break

    temp_file= "temp_chunk.wav"
    sf.write(temp_file, chunk,sr)

    start= time.time()
    result = model.transcribe(temp_file)
    end =  time.time()

    partial_text = " ".join(result["text"]).strip()

    full_transcript += " "+ partial_text

    print(f"[Chunk {i//chunk_size + 1}]")
    print("Partial:", partial_text)
    print("Accumulated:", full_transcript.strip())
    print("Latency:", round(end - start, 2), "seconds")
    print("-" * 40)

print("\nFinal Transcript")
print(full_transcript.strip())
    
