import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import librosa

duration=5
device_id=7
device_info= sd.query_devices(device_id)
fs = int(device_info['default_samplerate'])

print("recording")

recording = sd.rec(int(duration*fs),samplerate=fs,channels=1,device=device_id)
sd.wait()

print("resampling to 16000 Hz")

audio=recording.flatten()

audio_resampled = librosa.resample(audio,orig_sr=fs, target_sr=16000)
audio_int16= (audio_resampled*32767). astype(np.int16)

write("audio.wav", 16000,audio_int16)
print("saved")