import serial
import time
import wave
import numpy as np
import sounddevice as sd
import os
import pyttsx3
from openai import OpenAI
from faster_whisper import WhisperModel


# -----------------------------
# SETTINGS
# -----------------------------

COM_PORT = "COM7"
BAUD_RATE = 115200

SAMPLE_RATE = 16000
RECORD_SECONDS = 8

WAV_FILENAME = "tibby_recording.wav"

# AI settings (same service the team's tibby_central.py uses)
AI_BASE_URL = "https://api.ai.it.ufl.edu/v1/"
AI_MODEL = "gpt-oss-20b"
AI_INSTRUCTIONS = "You are TIBBY, a helpful voice assistant. Respond in at most 2 sentences."

# The key is read from your computer's settings instead of being typed
# into this file, so it never ends up on GitHub by accident.
AI_API_KEY = os.environ.get("TIBBY_AI_KEY", "")


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

    if not AI_API_KEY:
        print()
        print("No AI key found (TIBBY_AI_KEY is not set).")
        return "I heard you say: " + transcription

    print()
    print("Asking the AI...")

    try:
        client = OpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL)

        response = client.responses.create(
            model=AI_MODEL,
            instructions=AI_INSTRUCTIONS,
            input=transcription,
        )

        answer = response.output_text.strip()

        # Clean up the stored copy on the server, like tibby_central.py does
        try:
            client.responses.delete(response.id)
        except Exception:
            pass

        if answer:
            return answer

        return "Sorry, I didn't get an answer."

    except Exception as error:
        print()
        print("AI request failed:")
        print(error)
        return "Sorry, I couldn't reach the AI right now."


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

def open_serial():
    """Opens COM7, retrying until the board is plugged in and ready."""
    while True:
        try:
            ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
            time.sleep(1)
            return ser
        except serial.SerialException:
            print(f"Waiting for TIBBY on {COM_PORT}... (is it plugged in?)")
            time.sleep(2)


print("Starting TIBBY Button Demo...")
print(f"Connecting to TIBBY on {COM_PORT}...")

ser = open_serial()

print("Connected.")
print()
print("Press the TIBBY button to record.")
print()


while True:

    try:
        line = ser.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()
    except serial.SerialException:
        # The board disconnected (for example after pressing reset).
        print()
        print("TIBBY disconnected - reconnecting...")
        try:
            ser.close()
        except Exception:
            pass
        time.sleep(2)
        ser = open_serial()
        print("Reconnected.")
        print()
        continue

    if line:

        print("ESP32:", line)

        if "button pressed" in line.lower():

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

            # Ignore anything the board sent while we were busy
            ser.reset_input_buffer()
