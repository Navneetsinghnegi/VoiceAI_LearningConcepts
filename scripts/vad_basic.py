import librosa
import numpy as np
import soundfile as sf
import whisper
import os

model = whisper.load_model("base")

audio_path = r"D:\VoiceAI_TUT\422-122949-0000.flac"
audio, sr = librosa.load(audio_path,sr=16000)

frame_duration = 0.02
frame_size = int(frame_duration*sr)
energy_threshold = 0.001
silence_tolerance = int(0.3 / frame_duration)
min_speech_frames = int(0.3 / frame_duration)

speech_segment = []
current_segment = []
is_speaking = False
silence_counter = 0
speech_frame_count = 0

print("Running VAD")

for i in range(0, len(audio), frame_size):
    frame = audio[i:i + frame_size]

    if len(frame) < frame_size:
        break

    energy = np.mean(frame ** 2)

    if energy > energy_threshold:
        if not is_speaking:
            is_speaking = True
            current_segment = []
            speech_frame_count = 0

        current_segment.extend(frame)
        speech_frame_count += 1
        silence_counter = 0

    else:
        if is_speaking:
            silence_counter += 1

            if silence_counter < silence_tolerance:
                current_segment.extend(frame)
            else:
                if speech_frame_count >= min_speech_frames:
                    speech_segment.append(np.array(current_segment))
                is_speaking = False
                silence_counter = 0

print(f"Detected {len(speech_segment)} speech segments\n")

full_transcript = ""

for idx, segment in enumerate(speech_segment):
    

    result = model.transcribe(segment)
    text = " ".join(result["text"]).strip()

    full_transcript+= " "+text

print("\nFinal Transcript:")
print(full_transcript.strip())
