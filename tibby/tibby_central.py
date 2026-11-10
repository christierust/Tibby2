import asyncio
import struct
import time
import wave
import numpy as np
from bleak import BleakScanner, BleakClient
import matplotlib.pyplot as plt
from faster_whisper import WhisperModel
from openai import OpenAI
import pyttsx3

BT_BASE_UUID = "0000{:04x}-0000-1000-8000-00805f9b34fb"
HEART_RATE_CHR = 0x2A37
SAMPLE_RATE = 16000
TARGET_SAMPLES = 1000 * 256  # 2560 samples

def get_chatgpt_response(user_text: str) -> str:
    api_key = "sk-4h4pjnTT5Im3Wa9idCEchw"
    base_url = "https://api.ai.it.ufl.edu/v1/"

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.responses.create(
        model="gpt-5-nano",
        instructions="Respond in at most 2 sentences.",
        input=user_text,
    )

    response_id = response.id

    retrieved_response = client.responses.retrieve(response_id)
    print(f"Response text is: {retrieved_response.output_text}")
    delete_response = client.responses.delete(response_id)
    return retrieved_response.output_text.strip()

async def main():
    device = await BleakScanner.find_device_by_name("TIBBY_peripheral", timeout=5.0)
    if device is None:
        print("Timeout: no device found in 5 seconds.")
        return

    all_samples = []
    total_bytes = 0
    done_event = asyncio.Event()

    def notification_handler(sender, data):
        """Called each time the peripheral sends a notification."""
        nonlocal total_bytes
        num = len(data) // 2
        samples = struct.unpack(f"<{num}h", data[:num * 2])
        all_samples.extend(samples)
        total_bytes += len(data)
        print(f"Notification: {num} samples (total: {len(all_samples)})")
        if len(all_samples) >= TARGET_SAMPLES:
            done_event.set()

    async with BleakClient(device) as client:
        print(f"Connected to {device.name}, MTU: {client.mtu_size}")
        print(f"Subscribing to notifications, target: {TARGET_SAMPLES} samples...")

        start_time = time.perf_counter()
        await client.start_notify(
            BT_BASE_UUID.format(HEART_RATE_CHR), notification_handler
        )

        # Wait until we have enough samples or 30s timeout
        try:
            await asyncio.wait_for(done_event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            print("Timed out waiting for data.")

        try:
            await client.stop_notify(BT_BASE_UUID.format(HEART_RATE_CHR))
        except Exception as e:
            print(f"Warning: stop_notify failed ({e}), continuing anyway...")
        elapsed = time.perf_counter() - start_time

        # Trim to target
        all_samples = all_samples[:TARGET_SAMPLES]
        print(f"\nCollected {len(all_samples)} samples in {elapsed:.4f}s")
        print(f"Throughput: {total_bytes / elapsed:.0f} bytes/s ({total_bytes / elapsed / 1024:.2f} KB/s)")
        print(f"Sample rate achieved: {len(all_samples) / elapsed:.0f} samples/s")
        print(f"First 10: {all_samples[:10]}")
        print(f"Last 10: {all_samples[-10:]}")

        # Save audio as WAV
        audio = np.array(all_samples, dtype=np.int16)
        wav_filename = "tibby_recording.wav"
        with wave.open(wav_filename, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio.tobytes())
        print(f"Saved {len(audio)} samples to {wav_filename}")

        # Speech-to-Text using local Whisper (free, offline)
        print("Transcribing audio (local Whisper)...")
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, info = model.transcribe(wav_filename)
        transcription = " ".join(segment.text.strip() for segment in segments)
        print(f"Transcription: {transcription}")
        if transcription.strip():
            print("Sending transcription to the AI model")
            try:
                chatgpt_response = get_chatgpt_response(transcription)
                print(f"Response:\n {chatgpt_response}\n")
    
                print("Speaking the response (pyttsx3)...")
                response_tts = pyttsx3.init()
                response_tts.setProperty("rate", 200)
                response_tts.setProperty("volume", 1.0)
                response_tts.say(chatgpt_response)
                response_tts.runAndWait()
                print("Finished playing the response.")
            except Exception as e:
                print(f"ChatGPT request failed: {e}")
        else:
            print("No transcription text was produced, skipping ChatGPT call.")

asyncio.run(main())