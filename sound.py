import pyaudio
import wave

chunk = 1024
format = pyaudio.paInt16
channels = 1
rate = 44100
seconds = 5
filename = "test.wav"

p = pyaudio.PyAudio()

stream = p.open(format=format,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=chunk)

print("Recording...")

frames = []

for i in range(0, int(rate / chunk * seconds)):
    data = stream.read(chunk, exception_on_overflow=False)
    frames.append(data)

print("Finished recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(format))
wf.setframerate(rate)
wf.writeframes(b''.join(frames))
wf.close()

print("Saved test.wav")