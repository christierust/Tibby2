"""
TIBBY Bluetooth Button Demo - press to start, press to stop
-----------------------------------------------------------
Wireless demo (works with the board on battery):

  1. Press B1 once: recording starts on the LAPTOP microphone.
  2. Ask your question - take as long as you need.
  3. Press B1 again: recording stops.
  4. Whisper turns it into text, the UF AI answers, and the laptop speaks it.

Recording also stops on its own after MAX_RECORD_SECONDS, in case the
second press is missed.

Needs the updated main.c (the one that sends "BTN") flashed to the board.

Run it with:   python tibby_button_demo_toggle.py
Stop it with:  Ctrl+C
"""

import asyncio
import os
import queue
import threading
import time
import wave

import numpy as np
import pyttsx3
import sounddevice as sd
from bleak import BleakClient, BleakScanner
from faster_whisper import WhisperModel
from openai import OpenAI


# -----------------------------
# SETTINGS
# -----------------------------

DEVICE_NAME = "TIBBY_peripheral"
BOARD_ADDRESS = "E8:3D:C1:F5:47:72"          # backup, if the name isn't found
STREAM_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
BUTTON_MESSAGE = b"BTN"

SAMPLE_RATE = 16000
MAX_RECORD_SECONDS = 30      # safety limit if the second press is missed
MIN_RECORD_SECONDS = 1.0     # ignore a "stop" press sooner than this
WAV_FILENAME = "tibby_recording.wav"

AI_BASE_URL = "https://api.ai.it.ufl.edu/v1/"
AI_MODEL = "gpt-oss-20b"
AI_INSTRUCTIONS = ("You are TIBBY, a helpful voice assistant. "
                   "Respond in at most 2 sentences.")
AI_API_KEY = os.environ.get("TIBBY_AI_KEY", "").strip()


# -----------------------------
# LOAD SPEECH-TO-TEXT ONCE
# -----------------------------

print("Starting TIBBY Bluetooth Button Demo...")
print("Loading speech-to-text model (the first time can take a minute)...")
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")


# -----------------------------
# RECORD FROM LAPTOP MICROPHONE
# -----------------------------

def record_until_pressed():
    """Records from the laptop mic until B1 is pressed again (or the limit)."""
    chunks = []

    def on_audio(indata, frames, time_info, status):
        chunks.append(indata.copy())

    print()
    print("Recording started - speak now!")
    print("Press B1 again when you're done.")

    start = time.time()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="int16", callback=on_audio):
        while True:
            elapsed = time.time() - start
            if elapsed >= MAX_RECORD_SECONDS:
                print(f"Reached {MAX_RECORD_SECONDS} seconds - stopping automatically.")
                break
            try:
                button_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if elapsed < MIN_RECORD_SECONDS:
                continue   # too soon - probably the same press, ignore it
            print("TIBBY: button pressed - stopping")
            break

    seconds = time.time() - start
    if not chunks:
        print("No audio was captured.")
        return False

    audio = np.concatenate(chunks).squeeze()
    with wave.open(WAV_FILENAME, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    print(f"Recording finished ({seconds:.1f} seconds).")
    return True


# -----------------------------
# SPEECH TO TEXT
# -----------------------------

def transcribe_audio():
    print("Transcribing...")
    segments, info = whisper_model.transcribe(WAV_FILENAME)
    transcription = " ".join(segment.text.strip() for segment in segments)

    print()
    print("You said:")
    print(transcription)
    return transcription


# -----------------------------
# ASK THE AI
# -----------------------------

def ask_ai(question):
    if not AI_API_KEY:
        print()
        print("No AI key found (TIBBY_AI_KEY is not set).")
        return "I heard you say: " + question

    print()
    print("Asking the AI...")
    client = OpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL)

    # First try the same style of request as tibby_central.py
    try:
        response = client.responses.create(
            model=AI_MODEL,
            instructions=AI_INSTRUCTIONS,
            input=question,
        )
        answer = (response.output_text or "").strip()
        try:
            client.responses.delete(response.id)
        except Exception:
            pass
        if answer:
            return answer
    except Exception as first_error:
        print(f"(First AI request style failed: {first_error})")

    # Backup: the more common "chat" style of request
    try:
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": AI_INSTRUCTIONS},
                {"role": "user", "content": question},
            ],
        )
        answer = (response.choices[0].message.content or "").strip()
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


def answer_question():
    question = transcribe_audio()
    if question.strip():
        speak_response(ask_ai(question))
    else:
        print("No speech detected.")


# -----------------------------
# BLUETOOTH (runs in its own background thread)
# -----------------------------
# On Windows, the microphone and speech libraries set up the main thread in a
# way the Bluetooth library can't share, so Bluetooth gets its own thread.
# It just puts a note in button_queue each time the board sends "BTN".

button_queue = queue.Queue()
busy = threading.Event()


async def find_board():
    print()
    print(f"Looking for {DEVICE_NAME} over Bluetooth...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10)
    if device is None:
        device = await BleakScanner.find_device_by_address(BOARD_ADDRESS, timeout=10)
    return device


async def run_session(device):
    disconnected = asyncio.Event()

    def on_disconnect(client):
        disconnected.set()

    def on_message(sender, data):
        # Audio chunks are ~250 bytes; the button message is just "BTN"
        if len(data) < 16 and bytes(data[:3]) == BUTTON_MESSAGE and not busy.is_set():
            button_queue.put(time.time())

    async with BleakClient(device, disconnected_callback=on_disconnect) as client:
        await client.start_notify(STREAM_UUID, on_message)

        print(f"Connected to {device.name or device.address}.")
        print()
        print("Press the TIBBY button (B1) to start recording.")
        print()

        await disconnected.wait()


async def bluetooth_main():
    while True:
        device = await find_board()

        if device is None:
            print("Could not find the board. Check that it is powered on")
            print("and that the laptop's Bluetooth is on. Trying again...")
            await asyncio.sleep(5)
            continue

        try:
            await run_session(device)
        except Exception as error:
            print(f"Bluetooth error: {error}")

        print("TIBBY disconnected - reconnecting...")
        await asyncio.sleep(3)


def bluetooth_thread():
    asyncio.run(bluetooth_main())


# -----------------------------
# MAIN PROGRAM
# -----------------------------

def main():
    threading.Thread(target=bluetooth_thread, daemon=True).start()

    while True:
        try:
            button_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        print("TIBBY: button pressed (over Bluetooth)")

        # Record until the second press
        got_audio = record_until_pressed()

        # While thinking and speaking, ignore button presses
        busy.set()
        if got_audio:
            answer_question()

        # Throw away any presses that happened while we were busy
        time.sleep(0.5)
        while not button_queue.empty():
            button_queue.get_nowait()
        busy.clear()

        print()
        print("----------------------------")
        print("Press B1 to ask another")
        print("question.")
        print("----------------------------")
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Stopped.")
