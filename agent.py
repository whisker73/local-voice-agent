import os
import io
import json
import httpx
import soundfile as sf
import sounddevice as sd
import speech_recognition as sr
from faster_whisper import WhisperModel
import ollama

# --- KONFIGURATION ---
DEVICE = "cuda"
VOXTRAL_URL = "http://localhost:8000/v1/audio/speech"
MODEL_NAME = "qwen2.5:7b"
# API-Key am besten im System setzen, z.B. per Terminal: export SERPER_API_KEY="dein-key"
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "DEIN_API_KEY_HIER")

print("Lade Faster-Whisper (STT)...")
stt_model = WhisperModel("large-v3", device=DEVICE, compute_type="int8_float16")

# --- TOOLS ---
def web_search(query: str) -> str:
    """Sucht mit der Serper Google API."""
    print(f"-> Agent sucht im Web: {query}")
    if SERPER_API_KEY == "DEIN_API_KEY_HIER":
        return "Systemhinweis: Websuche fehlgeschlagen, da kein gültiger API-Key konfiguriert ist."

    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    try:
        response = httpx.post(url, headers=headers, data=payload, timeout=10.0)
        response.raise_for_status()
        results = response.json()
        return str(results.get("organic", [])[:3])
    except Exception as e:
        return f"Fehler bei der Websuche: {e}"

def read_local_file(filepath: str) -> str:
    """Liest eine Textdatei aus dem Dokumente-Ordner aus."""
    print(f"-> Agent liest Datei: {filepath}")
    base_path = os.path.expanduser("~/Documents")
    path = os.path.abspath(os.path.join(base_path, filepath))

    # Sicherheitscheck: Verhindert Zugriff außerhalb des Documents-Ordners (Path Traversal)
    if not path.startswith(os.path.abspath(base_path)):
        return "Zugriff verweigert: Pfad liegt außerhalb des erlaubten Verzeichnisses."

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Fehler beim Lesen der Datei: {e}"
    return "Datei nicht gefunden."

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Suche im Internet nach aktuellen Informationen",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_local_file",
            "description": "Lies eine Datei aus dem Dokumente-Ordner",
            "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}}}
        }
    }
]

# --- AGENT LOGIK ---
def get_llm_response(prompt: str, messages: list) -> str:
    if prompt:
        messages.append({'role': 'user', 'content': prompt})

    # Iterative Tool-Call Logik verhindert unendliche Rekursionen
    while True:
        try:
            response = ollama.chat(model=MODEL_NAME, messages=messages, tools=tools)
        except Exception as e:
            return f"Es gab ein Problem mit dem Sprachmodell: {e}"

        messages.append(response.message)

        # Wenn keine Tools aufgerufen wurden, ist die finale Antwort fertig
        if not response.message.tool_calls:
            return response.message.content

        # Tool-Aufrufe abarbeiten
        for tool_call in response.message.tool_calls:
            name = tool_call.function.name
            args = tool_call.function.arguments

            if name == "web_search":
                result = web_search(**args)
            elif name == "read_local_file":
                result = read_local_file(**args)
            else:
                result = f"Unbekanntes Tool: {name}"

            messages.append({'role': 'tool', 'content': str(result), 'name': name})

def listen_and_transcribe(recognizer, source) -> str:
    print("\n[Bereit] Ich höre zu...")
    try:
        audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=15)
    except sr.WaitTimeoutError:
        return ""

    # In-Memory Verarbeitung statt Festplatten I/O (temp.wav)
    wav_io = io.BytesIO(audio_data.get_wav_data())
    segments, _ = stt_model.transcribe(wav_io, beam_size=5, language="de", vad_filter=True)
    text = "".join([s.text for s in segments]).strip()

    if text:
        print(f"User:  {text}")
    return text if len(text) > 2 else ""

def speak(text: str):
    if not text: return
    print(f"Agent: {text}")
    payload = {"input": text, "model": "mistralai/Voxtral-4B-TTS-2603", "response_format": "wav", "voice": "de_female"}
    try:
        response = httpx.post(VOXTRAL_URL, json=payload, timeout=120.0)
        response.raise_for_status()
        audio_array, sr_rate = sf.read(io.BytesIO(response.content), dtype="float32")
        sd.play(audio_array, sr_rate)
        sd.wait()
    except Exception as e:
        print(f"Audio-Fehler (TTS): {e}")

if __name__ == "__main__":
    print("\n=== Voxtral-Agent gestartet ===")
    messages = [{'role': 'system', 'content': 'Du bist ein hilfreicher KI-Assistent. Nutze deine Tools wenn nötig. Antworte kurz, bündig und in natürlichem Deutsch.'}]

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Kalibriere Mikrofon für Hintergrundgeräusche...")
            recognizer.adjust_for_ambient_noise(source, duration=2)

            while True:
                user_text = listen_and_transcribe(recognizer, source)
                if not user_text:
                    continue
                if "beenden" in user_text.lower():
                    speak("Ich beende mich nun. Auf Wiedersehen!")
                    break

                ai_response = get_llm_response(user_text, messages)
                speak(ai_response)
    except KeyboardInterrupt:
        print("\nAgent durch Benutzer beendet.")
