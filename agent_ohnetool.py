import os
import io
import httpx
import soundfile as sf
import sounddevice as sd
import speech_recognition as sr
from faster_whisper import WhisperModel
import ollama

# 1. Konfiguration
device = "cuda"
print("Lade Faster-Whisper (STT)...")
stt_model = WhisperModel("large-v3", device=device, compute_type="int8_float16")

def listen_and_transcribe():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n[Bereit] Ich höre zu...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        # Kurzes Timeout, damit er nicht ewig auf Stille lauscht
        try:
            audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=15)
        except sr.WaitTimeoutError:
            return "" # Nichts gesagt

    print("[Verarbeitung] Stimme erkannt, transkribiere...")
    wav_data = audio_data.get_wav_data()
    with open("temp.wav", "wb") as f:
        f.write(wav_data)

    # VAD-Filter aktiviert (Voice Activity Detection filtert pure Stille aus)
    segments, info = stt_model.transcribe(
        "temp.wav",
        beam_size=5,
        language="de",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    text = "".join([segment.text for segment in segments]).strip()

    if os.path.exists("temp.wav"):
        os.remove("temp.wav")

    # Halluzinations-Blacklist
    blacklist = [
        "vielen dank.", "vielen dank", "vielen dank!",
        "vielen dank fürs zuschauen.", "vielen dank fürs zuschauen",
        "untertitelung", "untertitel"
    ]

    # Ignoriere den Text, wenn er auf der Blacklist steht oder zu kurz ist
    if text.lower() in blacklist or len(text) <= 2:
        return ""

    return text

def get_llm_response(prompt):
    print(f"Du: {prompt}")
    response = ollama.chat(model='qwen2.5:7b', messages=[
        {'role': 'system', 'content': 'Du bist ein fachlich versierter, persönlicher Assistent. Antworte in flüssigem, natürlichem Deutsch. Nutze kurze Sätze, vermeide unnötige Füllwörter und geh direkt auf den Punkt ein.'},
        {'role': 'user', 'content': prompt},
    ])
    return response['message']['content']

def speak(text):
    print(f"Agent: {text}")
    BASE_URL = "http://localhost:8000/v1"

    payload = {
        "input": text,
        "model": "mistralai/Voxtral-4B-TTS-2603",
        "response_format": "wav",
        "voice": "de_male", # <--- HIER DIE STIMME EINTRAGEN
    }

    try:
        response = httpx.post(f"{BASE_URL}/audio/speech", json=payload, timeout=120.0)
        response.raise_for_status()

        audio_array, sr_rate = sf.read(io.BytesIO(response.content), dtype="float32")
        sd.play(audio_array, sr_rate)
        sd.wait()
    except Exception as e:
        print(f"Fehler bei der Audio-Generierung: {e}")

# Hauptschleife
if __name__ == "__main__":
    print("\n=== Voxtral Sprachassistent gestartet ===")
    try:
        while True:
            user_text = listen_and_transcribe()

            # Wenn nichts Sinnvolles erkannt wurde (Blacklist oder Stille), springe sofort zurück
            if not user_text:
                continue

            if "beenden" in user_text.lower() or "tschüss" in user_text.lower():
                speak("Bis zum nächsten Mal!")
                break

            ai_response = get_llm_response(user_text)
            speak(ai_response)

    except KeyboardInterrupt:
        print("\nAgent manuell beendet.")
