import whisper
import speech_recognition as sr
import numpy as np
import torch
model = whisper.load_model("base")
recognizer = sr.Recognizer()

try:
    with sr.Microphone(sample_rate=16000) as source:
        print("\nListening...")

        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source, duration=0.5)

        # Record audio
        audio = recognizer.listen(
            source,
            timeout=10,  # Maximum time to wait for speech
            phrase_time_limit=30  # Maximum time for speech
        )
        print("Recording complete.")
except Exception as e:
    print(f"Error recording audio: {str(e)}")

try:

    audio_data = np.frombuffer(
        audio.get_raw_data(),
        np.int16
    ).flatten()

    # Convert to float32 and normalize
    float_audio = audio_data.astype(np.float32) / 32768.0

except Exception as e:
    print(f"Error preprocessing audio: {str(e)}")

try:
    result = model.transcribe(
                    float_audio,
                    language='en',  # Specify language or 'None' for auto-detection
                    fp16=False,     # Use FP16 for faster inference if GPU available
                    task="transcribe"  # Can be "transcribe" or "translate"
                )

    print(result["text"].strip())

except Exception :
    print("error in transcribing")
