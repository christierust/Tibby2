import wave
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
RECORD_SECONDS = 8
WAV_FILENAME = "tibby_recording.wav"

print("TIBBY PC Microphone Demo")
print("Speak when recording starts...")

# Record from the computer's default microphone
audio = sd.rec(
    int(RECORD_SECONDS * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16"
)

sd.wait()

print("Recording finished.")

# Convert to a simple 1-D array
audio = np.squeeze(audio)

# Save recording as a WAV file
with wave.open(WAV_FILENAME, "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(audio.tobytes())

print(f"Saved recording to {WAV_FILENAME}")

# Run local Whisper
print("Transcribing audio...")

model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

segments, info = model.transcribe(WAV_FILENAME)

transcription = " ".join(
    segment.text.strip()
    for segment in segments
)

print()
print("Transcription:")
print(transcription)
import pyttsx3

if transcription.strip():
    # Temporary demo response until the AI API key is fixed
    response = "I heard you say: " + transcription

    print()
    print("TIBBY Response:")
    print(response)

    print("Speaking response...")

    engine = pyttsx3.init()
    engine.setProperty("rate", 200)
    engine.setProperty("volume", 1.0)

    engine.say(response)
    engine.runAndWait()

    print("Finished.")
else:
    print("No speech was detected.")