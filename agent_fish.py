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
print("Lade Faster-Whisper (Gehör)...")
stt_model = WhisperModel("large-v3", device=device, compute_type="int8_float16")

# 2. Referenz-Audio automatisch analysieren (DER FIX)
print("Analysiere deine Stimm-Vorlage...")
if not os.path.exists("referenz.wav"):
    print("FEHLER: Die Datei 'referenz.wav' fehlt im Ordner.")
    exit(1)

ref_segments, _ = stt_model.transcribe("referenz.wav", beam_size=5, language="de")
reference_transcript = "".join([segment.text for segment in ref_segments]).strip()
print(f"Erkannter Text in der Vorlage: '{reference_transcript}'")

def listen_and_transcribe():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n[Bereit] Ich höre zu...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=15)
        except sr.WaitTimeoutError:
            return ""

    print("[Verarbeitung] Stimme erkannt, transkribiere...")
    wav_data = audio_data.get_wav_data()
    with open("temp.wav", "wb") as f:
        f.write(wav_data)

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

    blacklist = [
        "vielen dank.", "vielen dank", "vielen dank!",
        "vielen dank fürs zuschauen.", "vielen dank fürs zuschauen",
        "untertitelung", "untertitel"
    ]

    if text.lower() in blacklist or len(text) <= 2:
        return ""

    return text

def get_llm_response(prompt):
    print(f"Du: {prompt}")
    response = ollama.chat(model='qwen2.5:7b', messages=[
        {'role': 'system', 'content': 'Du bist ein lokaler KI-Assistent. Antworte immer auf Deutsch und halte deine Antworten prägnant (maximal 2-3 Sätze), da sie vorgelesen werden.'},
        {'role': 'user', 'content': prompt},
    ])
    return response['message']['content']

def speak(text):
    print(f"Agent: {text}")
    BASE_URL = "http://localhost:8000/v1"

    payload = {
        "input": text,
        "model": "fishaudio/s2-pro",
        "response_format": "wav",
        "voice": "holger_voice",
        "ref_text": reference_transcript  # <--- HIER IST DER FEHLENDE PARAMETER
    }

    try:
        response = httpx.post(f"{BASE_URL}/audio/speech", json=payload, timeout=120.0)
        response.raise_for_status()

        audio_array, sr_rate = sf.read(io.BytesIO(response.content), dtype="float32")
        sd.play(audio_array, sr_rate)
        sd.wait()
    except httpx.HTTPStatusError as e:
        print(f"\n--- SERVER FEHLER ---")
        print(f"Status: {e.response.status_code}")
        print(f"Details: {e.response.text}")
        print(f"---------------------\n")
    except Exception as e:
        print(f"Lokaler Fehler bei der Audio-Verarbeitung: {e}")

if __name__ == "__main__":
    print("\n=== Fish Speech Sprachassistent gestartet ===")
    try:
        while True:
            user_text = listen_and_transcribe()

            if not user_text:
                continue

            if "beenden" in user_text.lower() or "tschüss" in user_text.lower():
                speak("Bis zum nächsten Mal!")
                break

            ai_response = get_llm_response(user_text)
            speak(ai_response)

    except KeyboardInterrupt:
        print("\nAgent manuell beendet.")
