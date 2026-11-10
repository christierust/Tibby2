import serial
import time
import wave
import numpy as np
import sounddevice as sd
import pyttsx3
from faster_whisper import WhisperModel


# -----------------------------
# SETTINGS
# -----------------------------

COM_PORT = "COM7"
BAUD_RATE = 115200

SAMPLE_RATE = 16000
RECORD_SECONDS = 8

WAV_FILENAME = "tibby_recording.wav"


# -----------------------------
# RECORD FROM LAPTOP MICROPHONE
# -----------------------------

def record_audio():

    print()
    print("Recording started - speak now!")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    audio = np.squeeze(audio)

    with wave.open(WAV_FILENAME, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    print("Recording finished.")
    print(f"Saved to {WAV_FILENAME}")


# -----------------------------
# SPEECH TO TEXT
# -----------------------------

def transcribe_audio():

    print("Transcribing...")

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
    print("You said:")
    print(transcription)

    return transcription


# -----------------------------
# RESPONSE
# -----------------------------

def create_response(transcription):

    # Temporary response until your AI API credential is fixed
    return "I heard you say: " + transcription


# -----------------------------
# TEXT TO SPEECH
# -----------------------------

def speak_response(response):

    print()
    print("TIBBY response:")
    print(response)

    engine = pyttsx3.init()

    engine.setProperty("rate", 200)
    engine.setProperty("volume", 1.0)

    engine.say(response)
    engine.runAndWait()


# -----------------------------
# MAIN PROGRAM
# -----------------------------

print("Starting TIBBY Button Demo...")

print(f"Connecting to TIBBY on {COM_PORT}...")

ser = serial.Serial(
    COM_PORT,
    BAUD_RATE,
    timeout=1
)

time.sleep(2)

print("Connected.")
print()
print("Press the TIBBY button to record.")
print()


while True:

    line = ser.readline().decode(
        "utf-8",
        errors="ignore"
    ).strip()

    if line:

        print("ESP32:", line)

        if "recording started" in line:

            record_audio()

            transcription = transcribe_audio()

            if transcription.strip():

                response = create_response(
                    transcription
                )

                speak_response(response)

            else:

                print("No speech detected.")

            print()
            print("----------------------------")
            print("Press the TIBBY button again")
            print("for another recording.")
            print("----------------------------")
            print()