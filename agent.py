import os
import io
import logging
import httpx
import soundfile as sf
import sounddevice as sd
import speech_recognition as sr
from faster_whisper import WhisperModel
import ollama

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# --- KONFIGURATION ---
DEVICE = "cuda"
VOXTRAL_URL = "http://localhost:8000/v1/audio/speech"
MODEL_NAME = "qwen2.5:7b"
# API-Key per Terminal setzen: export SERPER_API_KEY="dein-key"
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")

# STT-Parameter
STT_LANGUAGE = "de"
STT_BEAM_SIZE = 5
LISTEN_TIMEOUT = 5
PHRASE_TIME_LIMIT = 15

# Agent-Grenzen
MAX_TOOL_ITERATIONS = 5   # Verhindert Endlosschleifen bei Tool-Calls
MAX_HISTORY_MESSAGES = 20  # Verhindert Kontext-Overflow

log.info("Lade Faster-Whisper (STT)...")
stt_model = WhisperModel("large-v3", device=DEVICE, compute_type="int8_float16")


# --- TOOLS ---
def web_search(query: str) -> str:
    """Sucht mit der Serper Google API und gibt formatierte Ergebnisse zurück."""
    log.info(f"Web-Suche: {query}")
    if not SERPER_API_KEY:
        return "Systemhinweis: Websuche nicht möglich – kein SERPER_API_KEY gesetzt."

    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    try:
        response = httpx.post(
            "https://google.serper.dev/search",
            headers=headers,
            json={"q": query},
            timeout=10.0,
        )
        response.raise_for_status()
        results = response.json().get("organic", [])[:3]
        if not results:
            return "Keine Suchergebnisse gefunden."
        # Formatiert für das LLM: Titel + Snippet + URL
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. {r.get('title', '(kein Titel)')}\n"
                f"   {r.get('snippet', '')}\n"
                f"   URL: {r.get('link', '')}"
            )
        return "\n\n".join(lines)
    except httpx.HTTPStatusError as e:
        return f"HTTP-Fehler bei der Websuche: {e.response.status_code}"
    except Exception as e:
        return f"Fehler bei der Websuche: {e}"


def read_local_file(filepath: str) -> str:
    """Liest eine Textdatei aus dem Dokumente-Ordner aus."""
    log.info(f"Lese Datei: {filepath}")
    base_path = os.path.abspath(os.path.expanduser("~/Documents"))
    path = os.path.abspath(os.path.join(base_path, filepath))

    # Sicherheitscheck: Verhindert Path Traversal
    if not path.startswith(base_path + os.sep):
        return "Zugriff verweigert: Pfad liegt außerhalb des erlaubten Verzeichnisses."

    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Datei nicht gefunden."
    except Exception as e:
        return f"Fehler beim Lesen der Datei: {e}"


# Tool-Registry: ermöglicht einfaches Hinzufügen neuer Tools
TOOL_REGISTRY: dict[str, callable] = {
    "web_search": web_search,
    "read_local_file": read_local_file,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Suche im Internet nach aktuellen Informationen zu einem Thema",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Der Suchbegriff"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_local_file",
            "description": "Lese eine Textdatei aus dem Dokumente-Ordner des Benutzers",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Relativer Pfad zur Datei innerhalb von ~/Documents",
                    }
                },
                "required": ["filepath"],
            },
        },
    },
]


# --- AGENT LOGIK ---
def trim_history(messages: list) -> list:
    """Kürzt die Konversationshistorie, um den Kontext nicht zu sprengen."""
    system_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "system"]
    other_msgs = [m for m in messages if not (isinstance(m, dict) and m.get("role") == "system")]
    if len(other_msgs) > MAX_HISTORY_MESSAGES:
        other_msgs = other_msgs[-MAX_HISTORY_MESSAGES:]
        log.debug("Konversationshistorie auf %d Nachrichten gekürzt.", MAX_HISTORY_MESSAGES)
    return system_msgs + other_msgs


def get_llm_response(prompt: str, messages: list) -> str:
    if prompt:
        messages.append({"role": "user", "content": prompt})

    for iteration in range(MAX_TOOL_ITERATIONS):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=trim_history(messages),
                tools=tools,
            )
        except Exception as e:
            return f"Es gab ein Problem mit dem Sprachmodell: {e}"

        messages.append(response.message)

        # Keine Tool-Calls → finale Antwort
        if not response.message.tool_calls:
            return response.message.content or ""

        # Tool-Aufrufe per Registry abarbeiten
        for tool_call in response.message.tool_calls:
            name = tool_call.function.name
            args = tool_call.function.arguments
            log.info("Tool-Aufruf [%d/%d]: %s(%s)", iteration + 1, MAX_TOOL_ITERATIONS, name, args)

            handler = TOOL_REGISTRY.get(name)
            result = handler(**args) if handler else f"Unbekanntes Tool: {name}"
            messages.append({"role": "tool", "content": str(result), "name": name})

    log.warning("Maximale Tool-Iterationen (%d) erreicht.", MAX_TOOL_ITERATIONS)
    return "Entschuldigung, ich konnte keine abschließende Antwort finden."


def listen_and_transcribe(recognizer: sr.Recognizer, source: sr.Microphone) -> str:
    log.info("Bereit – ich höre zu...")
    try:
        audio_data = recognizer.listen(
            source, timeout=LISTEN_TIMEOUT, phrase_time_limit=PHRASE_TIME_LIMIT
        )
    except sr.WaitTimeoutError:
        return ""

    # In-Memory-Verarbeitung, kein temporäres File auf der Festplatte
    wav_io = io.BytesIO(audio_data.get_wav_data())
    segments, _ = stt_model.transcribe(
        wav_io, beam_size=STT_BEAM_SIZE, language=STT_LANGUAGE, vad_filter=True
    )
    text = "".join(s.text for s in segments).strip()

    if text:
        log.info("User: %s", text)
    return text if len(text) > 2 else ""


def speak(text: str) -> None:
    if not text:
        return
    log.info("Agent: %s", text)
    payload = {
        "input": text,
        "model": "mistralai/Voxtral-4B-TTS-2603",
        "response_format": "wav",
        "voice": "de_female",
    }
    try:
        response = httpx.post(VOXTRAL_URL, json=payload, timeout=120.0)
        response.raise_for_status()
        audio_array, sr_rate = sf.read(io.BytesIO(response.content), dtype="float32")
        sd.play(audio_array, sr_rate)
        sd.wait()
    except httpx.HTTPStatusError as e:
        log.error("TTS-Fehler (HTTP %d): %s", e.response.status_code, e)
    except Exception as e:
        log.error("TTS-Fehler: %s", e)


if __name__ == "__main__":
    log.info("=== Voxtral-Agent gestartet ===")
    messages = [
        {
            "role": "system",
            "content": (
                "Du bist ein hilfreicher KI-Assistent. "
                "Nutze deine Tools, wenn du aktuelle oder lokale Informationen benötigst. "
                "Antworte kurz, bündig und in natürlichem Deutsch."
            ),
        }
    ]

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            log.info("Kalibriere Mikrofon für Hintergrundgeräusche...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            log.info("Kalibrierung abgeschlossen. Los geht's!")

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
        log.info("Agent durch Benutzer beendet.")
