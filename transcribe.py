import whisper
import time

model_size="base"

print("Loading model...")
model = whisper.load_model(model_size)

print("transcribing...")
start = time.time()

result = model.transcribe("./audio.wav")

end = time.time()

print("\nTranscription:")
print(result["text"])

print("\nTime Taken", round(end-start,2), "seconds")