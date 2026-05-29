import os
import io
import re
import sys
import json
import queue
import asyncio
import inspect
import logging
import argparse
import threading
import subprocess
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
from types import SimpleNamespace
from typing import Callable, get_type_hints
from urllib.parse import quote
from dotenv import load_dotenv
import httpx
import psutil
import soundfile as sf
import sounddevice as sd
import speech_recognition as sr
from faster_whisper import WhisperModel
import ollama
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Lädt .env Datei (SERPER_API_KEY etc.) – überschreibt keine gesetzten Env-Vars
load_dotenv()

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
# API-Keys in .env setzen (wird automatisch geladen) oder via: export KEY="wert"
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
HA_URL         = os.environ.get("HA_URL", "http://homeassistant.local:8123")
HA_TOKEN       = os.environ.get("HA_TOKEN", "")

# STT-Parameter
STT_LANGUAGE = "de"
STT_BEAM_SIZE = 5
LISTEN_TIMEOUT = 5
PHRASE_TIME_LIMIT = 15

# Agent-Grenzen
MAX_TOOL_ITERATIONS = 5   # Verhindert Endlosschleifen bei Tool-Calls
MAX_HISTORY_MESSAGES = 20  # Verhindert Kontext-Overflow

# Web-UI
UI_PORT = 7860

# Geteilter HTTP-Client mit Connection-Pooling (kein neuer TCP-Handshake pro Request)
_http = httpx.Client(headers={"User-Agent": "local-voice-agent/1.0"})

# --- WEB-UI ---
_ui_loop: asyncio.AbstractEventLoop | None = None
_ws_connections: list[WebSocket] = []


def emit(event_type: str, **kwargs) -> None:
    if not _ui_loop or not _ws_connections:
        return
    data = json.dumps({"type": event_type, **kwargs}, ensure_ascii=False)
    for ws in list(_ws_connections):
        asyncio.run_coroutine_threadsafe(_ws_send(ws, data), _ui_loop)


async def _ws_send(ws: WebSocket, data: str) -> None:
    try:
        await ws.send_text(data)
    except Exception:
        try:
            _ws_connections.remove(ws)
        except ValueError:
            pass


def _get_config() -> dict:
    return {
        "model_name": MODEL_NAME,
        "stt_language": STT_LANGUAGE,
        "max_tool_iterations": MAX_TOOL_ITERATIONS,
        "max_history_messages": MAX_HISTORY_MESSAGES,
    }


class _ConfigUpdate(BaseModel):
    model_name: str | None = None
    stt_language: str | None = None
    max_tool_iterations: int | None = None
    max_history_messages: int | None = None


_app = FastAPI()
_static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@_app.get("/")
async def _root():
    with open(os.path.join(_static_dir, "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


@_app.websocket("/ws")
async def _ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    _ws_connections.append(websocket)
    await websocket.send_text(json.dumps({"type": "config", **_get_config()}))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        try:
            _ws_connections.remove(websocket)
        except ValueError:
            pass


@_app.get("/api/config")
async def _api_get_config():
    return _get_config()


@_app.post("/api/config")
async def _api_post_config(update: _ConfigUpdate):
    global MODEL_NAME, STT_LANGUAGE, MAX_TOOL_ITERATIONS, MAX_HISTORY_MESSAGES
    if update.model_name is not None:
        MODEL_NAME = update.model_name
    if update.stt_language is not None:
        STT_LANGUAGE = update.stt_language
    if update.max_tool_iterations is not None:
        MAX_TOOL_ITERATIONS = update.max_tool_iterations
    if update.max_history_messages is not None:
        MAX_HISTORY_MESSAGES = update.max_history_messages
    cfg = _get_config()
    emit("config", **cfg)
    log.info("Konfiguration aktualisiert: %s", cfg)
    return cfg


def _start_ui_server() -> None:
    global _ui_loop
    _ui_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_ui_loop)
    config = uvicorn.Config(_app, host="0.0.0.0", port=UI_PORT, loop="none", log_level="warning")
    server = uvicorn.Server(config)
    _ui_loop.run_until_complete(server.serve())


# --- BARGE-IN ---
_audio_queue: queue.Queue = queue.Queue()
_is_speaking = threading.Event()
_barge_in_detected = threading.Event()

# Mindestlänge (Sekunden) für Barge-In – filtert kurze Geräusche heraus
BARGE_IN_MIN_DURATION = 0.8


def _audio_duration(audio: sr.AudioData) -> float:
    return len(audio.frame_data) / (audio.sample_rate * audio.sample_width)


def _bg_listener_loop(recognizer: sr.Recognizer, source: sr.Microphone) -> None:
    """Läuft permanent im Hintergrund, stellt Audio in _audio_queue.
    Barge-In wird nur ausgelöst wenn das Audio länger als BARGE_IN_MIN_DURATION ist."""
    log.info("Hintergrund-Listener aktiv.")
    while True:
        try:
            audio = recognizer.listen(source, timeout=1.0, phrase_time_limit=PHRASE_TIME_LIMIT)
            if _is_speaking.is_set():
                duration = _audio_duration(audio)
                if duration < BARGE_IN_MIN_DURATION:
                    log.debug("Kurzes Geräusch (%.2fs) während TTS – ignoriert.", duration)
                    continue  # weder queuen noch unterbrechen
                log.info("Barge-In erkannt (%.2fs) – stoppe TTS.", duration)
                _barge_in_detected.set()
                emit("status", state="listening")
                sd.stop()
            _audio_queue.put(audio)
        except sr.WaitTimeoutError:
            continue
        except Exception as e:
            log.warning("Hintergrund-Listener-Fehler: %s", e)


# Whitelist erlaubter Anwendungen (Konstante, nicht bei jedem Aufruf neu erstellt)
_ALLOWED_APPS = {
    "firefox", "chromium", "chromium-browser", "google-chrome",
    "nautilus", "thunar", "dolphin", "nemo",
    "gedit", "kate", "mousepad", "xed",
    "thunderbird", "evolution",
    "calculator", "gnome-calculator", "kcalc",
    "terminal", "gnome-terminal", "xterm", "konsole", "alacritty", "kitty",
    "vlc", "mpv", "rhythmbox", "clementine",
    "libreoffice", "libreoffice-writer", "libreoffice-calc",
    "code", "codium",
}

log.info("Lade Faster-Whisper (STT)...")
stt_model = WhisperModel("large-v3", device=DEVICE, compute_type="float16")


# --- TOOL-DECORATOR ---
# Bildet Python-Typen auf JSON-Schema-Typen ab
_PY_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

# Werden automatisch befüllt – nie manuell anfassen!
TOOL_REGISTRY: dict[str, Callable] = {}
tools: list[dict] = []


def tool(description: str, param_descriptions: dict[str, str] | None = None):
    """
    Decorator: registriert eine Funktion als LLM-Tool.

    Verwendung:
        @tool("Beschreibung des Tools", {"param": "Beschreibung des Parameters"})
        def mein_tool(param: str) -> str:
            ...

    - Erzeugt automatisch das Ollama-JSON-Schema aus den Typ-Hinweisen.
    - Pflichtparameter (ohne Default) werden automatisch in 'required' aufgenommen.
    - Registriert die Funktion in TOOL_REGISTRY und tools.
    """
    param_descriptions = param_descriptions or {}

    def decorator(func: Callable) -> Callable:
        hints = get_type_hints(func)
        hints.pop("return", None)
        sig = inspect.signature(func)

        properties: dict[str, dict] = {}
        required: list[str] = []

        for name, param in sig.parameters.items():
            json_type = _PY_TO_JSON.get(hints.get(name, str), "string")
            prop: dict = {"type": json_type}
            if name in param_descriptions:
                prop["description"] = param_descriptions[name]
            properties[name] = prop
            if param.default is inspect.Parameter.empty:
                required.append(name)

        schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

        TOOL_REGISTRY[func.__name__] = func
        tools.append(schema)
        log.debug("Tool registriert: %s", func.__name__)
        return func

    return decorator


# --- TOOLS ---
@tool(
    "Suche im Internet nach aktuellen Informationen zu einem Thema",
    {"query": "Der Suchbegriff"},
)
def web_search(query: str) -> str:
    """Sucht mit der Serper Google API und gibt formatierte Ergebnisse zurück."""
    log.info("Web-Suche: %s", query)
    if not SERPER_API_KEY:
        return "Systemhinweis: Websuche nicht möglich – kein SERPER_API_KEY gesetzt."

    try:
        response = _http.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query},
            timeout=10.0,
        )
        response.raise_for_status()
        results = response.json().get("organic", [])[:3]
        if not results:
            return "Keine Suchergebnisse gefunden."
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


@tool(
    "Lese eine Textdatei aus dem Dokumente-Ordner des Benutzers",
    {"filepath": "Relativer Pfad zur Datei innerhalb von ~/Documents"},
)
def read_local_file(filepath: str) -> str:
    """Liest eine Textdatei aus dem Dokumente-Ordner aus."""
    log.info("Lese Datei: %s", filepath)
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


@tool(
    "Schreibe oder erstelle eine Textdatei im Dokumente-Ordner",
    {
        "filepath": "Relativer Pfad zur Datei innerhalb von ~/Documents (z.B. 'notiz.txt')",
        "content": "Der zu schreibende Inhalt der Datei",
        "append": "Falls true, wird der Inhalt angehängt statt überschrieben (Standard: false)",
    },
)
def write_file(filepath: str, content: str, append: bool = False) -> str:
    """Schreibt Inhalt in eine Textdatei im Dokumente-Ordner."""
    log.info("Schreibe Datei: %s (append=%s)", filepath, append)
    base_path = os.path.abspath(os.path.expanduser("~/Documents"))
    path = os.path.abspath(os.path.join(base_path, filepath))

    if not path.startswith(base_path + os.sep):
        return "Zugriff verweigert: Pfad liegt außerhalb des erlaubten Verzeichnisses."

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)
        action = "angehängt" if append else "geschrieben"
        return f"Datei '{filepath}' erfolgreich {action}."
    except Exception as e:
        return f"Fehler beim Schreiben der Datei: {e}"


@tool(
    "Liste alle Dateien und Ordner im Dokumente-Ordner auf",
    {"subfolder": "Optionaler Unterordner innerhalb von ~/Documents (leer = Hauptordner)"},
)
def list_files(subfolder: str = "") -> str:
    """Listet den Inhalt des Dokumente-Ordners auf."""
    log.info("Liste Dateien: %s", subfolder or "~/Documents")
    base_path = os.path.abspath(os.path.expanduser("~/Documents"))
    target = os.path.abspath(os.path.join(base_path, subfolder)) if subfolder else base_path

    # os.sep verhindert, dass "/Documents2" als Unterordner von "/Documents" gilt
    if not (target == base_path or target.startswith(base_path + os.sep)):
        return "Zugriff verweigert."
    if not os.path.isdir(target):
        return f"Ordner '{subfolder}' nicht gefunden."

    try:
        entries = sorted(os.listdir(target))
        if not entries:
            return "Der Ordner ist leer."
        lines = []
        for e in entries:
            full = os.path.join(target, e)
            marker = "📁" if os.path.isdir(full) else "📄"
            lines.append(f"{marker} {e}")
        return "\n".join(lines)
    except Exception as e:
        return f"Fehler beim Auflisten: {e}"


@tool("Gibt das aktuelle Datum und die Uhrzeit zurück", {})
def get_datetime() -> str:
    """Gibt Datum und Uhrzeit in lesbarem Format zurück."""
    return datetime.now().strftime("Heute ist %A, der %d. %B %Y. Es ist %H:%M Uhr.")


@tool(
    "Berechne einen mathematischen Ausdruck",
    {"expression": "Der mathematische Ausdruck als Text, z.B. '15% von 340' oder '(12 * 8) + 5'"},
)
def calculate(expression: str) -> str:
    """Wertet einen mathematischen Ausdruck sicher aus."""
    log.info("Berechnung: %s", expression)
    # sympy bleibt lazy-import – Startup würde sonst ~1-2s länger dauern
    from sympy import sympify, SympifyError

    # Prozent-Kurzform auflösen: "15% von 340" → "0.15 * 340"
    expression = re.sub(
        r"(\d+(?:\.\d+)?)\s*%\s*(?:von|of)\s*(\d+(?:\.\d+)?)",
        lambda m: str(float(m.group(1)) / 100 * float(m.group(2))),
        expression,
        flags=re.IGNORECASE,
    )
    # Einfaches "%" als Division durch 100 auflösen
    expression = re.sub(r"(\d+(?:\.\d+)?)\s*%", lambda m: str(float(m.group(1)) / 100), expression)

    try:
        result = sympify(expression)
        return f"Ergebnis: {result}"
    except SympifyError:
        return f"Konnte den Ausdruck nicht berechnen: '{expression}'"
    except Exception as e:
        return f"Fehler bei der Berechnung: {e}"


@tool(
    "Zeigt das aktuelle Wetter für einen Ort an",
    {"location": "Stadt oder Ort, z.B. 'Berlin' oder 'München'"},
)
def get_weather(location: str) -> str:
    """Ruft das aktuelle Wetter via wttr.in ab (kostenlos, kein API-Key)."""
    log.info("Wetter für: %s", location)
    try:
        response = _http.get(
            f"https://wttr.in/{quote(location)}",
            params={"format": "j1", "lang": "de"},
            timeout=8.0,
        )
        response.raise_for_status()
        data = response.json()
        current = data["current_condition"][0]
        area = data["nearest_area"][0]
        city = area["areaName"][0]["value"]
        country = area["country"][0]["value"]
        temp_c = current["temp_C"]
        feels_like = current["FeelsLikeC"]
        desc = current["lang_de"][0]["value"] if current.get("lang_de") else current["weatherDesc"][0]["value"]
        humidity = current["humidity"]
        wind_kmph = current["windspeedKmph"]
        return (
            f"Wetter in {city}, {country}:\n"
            f"  {desc}, {temp_c}°C (gefühlt {feels_like}°C)\n"
            f"  Luftfeuchtigkeit: {humidity}%, Wind: {wind_kmph} km/h"
        )
    except httpx.HTTPStatusError as e:
        return f"Wetterdienst nicht erreichbar (HTTP {e.response.status_code})."
    except Exception as e:
        return f"Fehler beim Abrufen des Wetters: {e}"


@tool("Zeigt aktuelle System-Auslastung (CPU, RAM, Festplatte)", {})
def get_system_info() -> str:
    """Gibt CPU-, Speicher- und Festplattenauslastung zurück."""
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    mem_used_gb = mem.used / 1e9
    mem_total_gb = mem.total / 1e9
    disk_used_gb = disk.used / 1e9
    disk_total_gb = disk.total / 1e9

    return (
        f"System-Auslastung:\n"
        f"  CPU:       {cpu:.1f}%\n"
        f"  RAM:       {mem_used_gb:.1f} / {mem_total_gb:.1f} GB ({mem.percent:.1f}%)\n"
        f"  Festplatte:{disk_used_gb:.1f} / {disk_total_gb:.1f} GB ({disk.percent:.1f}%)"
    )


@tool(
    "Suche einen Begriff auf Wikipedia und gib eine kurze Zusammenfassung zurück",
    {"query": "Der Suchbegriff oder Name, nach dem auf Wikipedia gesucht werden soll"},
)
def wiki_search(query: str) -> str:
    """Ruft die deutsche Wikipedia-Zusammenfassung für einen Begriff ab."""
    log.info("Wikipedia-Suche: %s", query)
    try:
        search_resp = _http.get(
            "https://de.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query,
                    "format": "json", "srlimit": 1},
            timeout=8.0,
        )
        search_resp.raise_for_status()
        results = search_resp.json().get("query", {}).get("search", [])
        if not results:
            return f"Kein Wikipedia-Eintrag für '{query}' gefunden."
        title = results[0]["title"]

        summary_resp = _http.get(
            f"https://de.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}",
            timeout=8.0,
        )
        summary_resp.raise_for_status()
        data = summary_resp.json()
        extract = data.get("extract", "Keine Zusammenfassung verfügbar.")
        # Auf ~3 Sätze kürzen für Voice-Ausgabe
        sentences = extract.split(". ")
        short = ". ".join(sentences[:3]) + ("." if len(sentences) > 3 else "")
        return f"{data.get('title', title)}: {short}"
    except httpx.HTTPStatusError as e:
        return f"Wikipedia nicht erreichbar (HTTP {e.response.status_code})."
    except Exception as e:
        return f"Fehler bei der Wikipedia-Suche: {e}"


@tool(
    "Übersetze einen Text in eine andere Sprache",
    {
        "text": "Der zu übersetzende Text",
        "target_lang": "Zielsprache als Sprachcode, z.B. 'en' für Englisch, 'fr' für Französisch, 'es' für Spanisch",
        "source_lang": "Quellsprache als Sprachcode (Standard: 'de' für Deutsch)",
    },
)
def translate_text(text: str, target_lang: str, source_lang: str = "de") -> str:
    """Übersetzt Text via MyMemory API (kostenlos, kein API-Key nötig)."""
    log.info("Übersetze '%s' von %s nach %s", text[:30], source_lang, target_lang)
    try:
        response = _http.get(
            "https://api.mymemory.translated.net/get",
            params={"q": text, "langpair": f"{source_lang}|{target_lang}"},
            timeout=8.0,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("responseStatus") != 200:
            return f"Übersetzung fehlgeschlagen: {data.get('responseDetails', 'Unbekannter Fehler')}"
        translated = data["responseData"]["translatedText"]
        return f"Übersetzung ({source_lang} → {target_lang}): {translated}"
    except httpx.HTTPStatusError as e:
        return f"Übersetzungsdienst nicht erreichbar (HTTP {e.response.status_code})."
    except Exception as e:
        return f"Fehler bei der Übersetzung: {e}"


_HTML_TAG_RE = re.compile(r"<[^>]+>")

@tool(
    "Rufe aktuelle Nachrichten ab",
    {"category": "Nachrichtenkategorie: 'inland', 'ausland', 'wirtschaft', 'sport', 'video' oder leer für Top-Nachrichten"},
)
def get_news(category: str = "") -> str:
    """Ruft aktuelle Nachrichten vom Tagesschau-RSS-Feed ab (kein API-Key nötig)."""
    category_map = {
        "inland":     "https://www.tagesschau.de/xml/rss2_inland/",
        "ausland":    "https://www.tagesschau.de/xml/rss2_ausland/",
        "wirtschaft": "https://www.tagesschau.de/xml/rss2_wirtschaft/",
        "sport":      "https://www.tagesschau.de/xml/rss2_sport/",
        "video":      "https://www.tagesschau.de/xml/rss2/",
    }
    url = category_map.get(category.lower(), "https://www.tagesschau.de/xml/rss2/")
    log.info("Nachrichten abrufen: %s", url)

    try:
        response = _http.get(url, timeout=8.0)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        items = root.findall(".//item")[:5]
        if not items:
            return "Keine Nachrichten gefunden."
        lines = []
        for item in items:
            title = item.findtext("title", "").strip()
            desc = _HTML_TAG_RE.sub("", item.findtext("description", "").strip())[:120]
            lines.append(f"• {title}: {desc}")
        return "Aktuelle Nachrichten:\n" + "\n\n".join(lines)
    except Exception as e:
        return f"Fehler beim Abrufen der Nachrichten: {e}"


@tool(
    "Hänge eine schnelle Notiz an die persönliche Notizdatei an",
    {"note": "Der Inhalt der Notiz"},
)
def quick_note(note: str) -> str:
    """Hängt eine Notiz mit Zeitstempel an ~/Documents/notizen.md an."""
    log.info("Notiz: %s", note[:50])
    notes_path = os.path.expanduser("~/Documents/notizen.md")
    try:
        os.makedirs(os.path.dirname(notes_path), exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"\n- [{timestamp}] {note}"
        with open(notes_path, "a", encoding="utf-8") as f:
            f.write(entry)
        return f"Notiz gespeichert: '{note}'"
    except Exception as e:
        return f"Fehler beim Speichern der Notiz: {e}"


@tool(
    "Öffne eine Anwendung oder ein Programm auf dem Computer",
    {"app": "Name oder Befehl der Anwendung, z.B. 'firefox', 'nautilus', 'gedit', 'thunderbird'"},
)
def open_application(app: str) -> str:
    """Startet eine Anwendung via subprocess."""
    log.info("Öffne Anwendung: %s", app)
    app_lower = app.lower().strip()
    if app_lower not in _ALLOWED_APPS:
        return (f"'{app}' ist nicht in der erlaubten Anwendungsliste. "
                f"Erlaubt: {', '.join(sorted(_ALLOWED_APPS))}")
    if not shutil.which(app_lower):
        return f"Anwendung '{app}' nicht gefunden oder nicht installiert."
    try:
        subprocess.Popen([app_lower], start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"'{app}' wurde gestartet."
    except Exception as e:
        return f"Fehler beim Starten von '{app}': {e}"


# --- HOME ASSISTANT TOOLS ---
def _ha_headers() -> dict:
    return {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}


@tool(
    "Frage den Zustand eines Home-Assistant-Geräts ab (z.B. Temperatur, ob ein Licht an ist)",
    {
        "entity_id": (
            "Die Entity-ID des Geräts, z.B. 'light.wohnzimmer', 'sensor.schlafzimmer_temperatur'. "
            "Leer lassen um alle verfügbaren Entitäten aufzulisten."
        ),
    },
)
def ha_get_state(entity_id: str = "") -> str:
    """Ruft den Zustand einer oder aller HA-Entitäten ab."""
    if not HA_TOKEN:
        return "Systemhinweis: HA_TOKEN nicht gesetzt – Home Assistant nicht konfiguriert."
    try:
        if entity_id:
            log.info("HA get_state: %s", entity_id)
            resp = _http.get(f"{HA_URL}/api/states/{entity_id}", headers=_ha_headers(), timeout=8.0)
            resp.raise_for_status()
            data = resp.json()
            state = data.get("state", "unbekannt")
            attrs = data.get("attributes", {})
            friendly = attrs.get("friendly_name", entity_id)
            # Nützliche Attribute je nach Gerätetyp
            extras = []
            for key in ("unit_of_measurement", "temperature", "current_temperature",
                        "brightness", "color_temp", "humidity", "battery"):
                if key in attrs:
                    extras.append(f"{key}: {attrs[key]}")
            detail = f" ({', '.join(extras)})" if extras else ""
            return f"{friendly}: {state}{detail}"
        else:
            log.info("HA get_state: alle Entitäten")
            resp = _http.get(f"{HA_URL}/api/states", headers=_ha_headers(), timeout=10.0)
            resp.raise_for_status()
            entities = resp.json()
            # Interne/uninteressante Domains ausblenden
            _SKIP_DOMAINS = {"automation", "script", "scene", "zone", "person",
                             "sun", "weather", "input_boolean", "input_number",
                             "input_select", "input_text", "timer", "counter",
                             "persistent_notification", "update"}
            groups: dict[str, list[str]] = {}
            for e in entities:
                domain = e["entity_id"].split(".")[0]
                if domain in _SKIP_DOMAINS:
                    continue
                entity_id = e["entity_id"]
                name = e.get("attributes", {}).get("friendly_name") or entity_id
                state = e["state"]
                groups.setdefault(domain, []).append(f"{name} ({entity_id}): {state}")
            if not groups:
                return "Keine Geräte gefunden."
            lines = []
            for domain in sorted(groups):
                lines.append(f"[{domain}]")
                lines.extend(f"  {item}" for item in groups[domain])
            return "\n".join(lines)
    except httpx.HTTPStatusError as e:
        return f"Home Assistant Fehler (HTTP {e.response.status_code}): {e.response.text[:200]}"
    except Exception as e:
        return f"Fehler bei HA-Abfrage: {e}"


@tool(
    "Steuere ein Home-Assistant-Gerät oder rufe einen Dienst auf (Licht an/aus, Temperatur setzen, Szene aktivieren …)",
    {
        "domain":    "Die Domäne des Dienstes, z.B. 'light', 'switch', 'climate', 'scene', 'automation', 'media_player'",
        "service":   "Der aufzurufende Dienst, z.B. 'turn_on', 'turn_off', 'toggle', 'set_temperature', 'turn_on' (für Szenen)",
        "entity_id": "Die Entity-ID des Zielgeräts, z.B. 'light.wohnzimmer' oder 'scene.abend'",
        "extra":     "Optionale JSON-Zusatzparameter als String, z.B. '{\"temperature\": 21}' oder '{\"brightness\": 128}'",
    },
)
def ha_call_service(domain: str, service: str, entity_id: str, extra: str = "") -> str:
    """Ruft einen Home-Assistant-Dienst auf."""
    if not HA_TOKEN:
        return "Systemhinweis: HA_TOKEN nicht gesetzt – Home Assistant nicht konfiguriert."
    log.info("HA call_service: %s.%s(%s)", domain, service, entity_id)
    payload: dict = {"entity_id": entity_id}
    if extra:
        try:
            payload.update(json.loads(extra))
        except json.JSONDecodeError:
            return f"Ungültiger JSON-Extra-Parameter: {extra}"
    try:
        resp = _http.post(
            f"{HA_URL}/api/services/{domain}/{service}",
            headers=_ha_headers(),
            json=payload,
            timeout=10.0,
        )
        resp.raise_for_status()
        changed = resp.json()
        if changed:
            names = [e.get("attributes", {}).get("friendly_name", e["entity_id"]) for e in changed]
            return f"Erledigt: {', '.join(names)}"
        return "Dienst ausgeführt."
    except httpx.HTTPStatusError as e:
        return f"Home Assistant Fehler (HTTP {e.response.status_code}): {e.response.text[:200]}"
    except Exception as e:
        return f"Fehler beim HA-Dienst-Aufruf: {e}"


# --- CLAUDE BACKEND ---
class ClaudeBackend:
    """Ruft die Claude Code CLI als Subprocess auf, behält die Session via --continue."""

    def __init__(self):
        self._first_call = True

    def chat(self, user_text: str) -> str:
        cmd = ["claude", "-p", user_text, "--output-format", "text"]
        if not self._first_call:
            cmd.append("--continue")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self._first_call = False
            if result.returncode != 0:
                log.error("Claude CLI Fehler: %s", result.stderr[:200])
                return "Es gab ein Problem mit Claude. Bitte versuche es erneut."
            return result.stdout.strip()
        except FileNotFoundError:
            return "claude-Befehl nicht gefunden. Ist Claude Code installiert?"
        except subprocess.TimeoutExpired:
            return "Claude hat zu lange gebraucht. Bitte versuche es erneut."
        except Exception as e:
            return f"Fehler beim Aufruf von Claude: {e}"


# --- AGENT LOGIK ---
def _parse_text_tool_calls(content: str) -> list:
    """Fallback: extrahiert Tool-Calls aus Klartext wenn das Modell sie nicht strukturiert liefert."""
    result = []
    for m in re.finditer(
        r'"name"\s*:\s*"(\w+)".*?"arguments"\s*:\s*(\{[^}]*\})',
        content, re.DOTALL
    ):
        name = m.group(1)
        if name not in TOOL_REGISTRY:
            continue
        try:
            args = json.loads(m.group(2))
            result.append(SimpleNamespace(function=SimpleNamespace(name=name, arguments=args)))
        except json.JSONDecodeError:
            continue
    return result


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
        emit("status", state="thinking")
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=trim_history(messages),
                tools=tools,
            )
        except Exception as e:
            return f"Es gab ein Problem mit dem Sprachmodell: {e}"

        messages.append(response.message)

        # Strukturierte Tool-Calls bevorzugen; Fallback auf Text-Parsing
        tool_calls = response.message.tool_calls
        if not tool_calls:
            text_calls = _parse_text_tool_calls(response.message.content or "")
            if not text_calls:
                return response.message.content or ""
            log.warning("Modell gab Tool-Call als Klartext aus – Fallback-Parser aktiv")
            tool_calls = text_calls

        # Tool-Aufrufe per Registry abarbeiten
        for tool_call in tool_calls:
            name = tool_call.function.name
            args = tool_call.function.arguments
            log.info("Tool-Aufruf [%d/%d]: %s(%s)", iteration + 1, MAX_TOOL_ITERATIONS, name, args)
            emit("tool_call", name=name, args=args if isinstance(args, dict) else {})

            handler = TOOL_REGISTRY.get(name)
            if handler:
                valid = set(inspect.signature(handler).parameters)
                args = {k: v for k, v in args.items() if k in valid}
                result = handler(**args)
            else:
                result = f"Unbekanntes Tool: {name}"
            emit("tool_result", name=name, result=str(result)[:600])
            messages.append({"role": "tool", "content": str(result), "name": name})

    log.warning("Maximale Tool-Iterationen (%d) erreicht.", MAX_TOOL_ITERATIONS)
    return "Entschuldigung, ich konnte keine abschließende Antwort finden."


def listen_and_transcribe() -> str:
    emit("status", state="listening")
    try:
        audio_data = _audio_queue.get(timeout=0.5)
    except queue.Empty:
        return ""

    emit("status", state="transcribing")
    wav_io = io.BytesIO(audio_data.get_wav_data())
    segments, _ = stt_model.transcribe(
        wav_io, beam_size=STT_BEAM_SIZE, language=STT_LANGUAGE, vad_filter=True
    )
    text = "".join(s.text for s in segments).strip()

    if text:
        log.info("User: %s", text)
        emit("stt", text=text)
    return text if len(text) > 1 else ""


def speak(text: str) -> None:
    if not text:
        return
    emit("response", text=text)
    emit("status", state="speaking")
    log.info("Agent: %s", text)
    payload = {
        "input": text,
        "model": "mistralai/Voxtral-4B-TTS-2603",
        "response_format": "wav",
        "voice": "de_female",
    }
    try:
        response = _http.post(VOXTRAL_URL, json=payload, timeout=120.0)
        response.raise_for_status()
        audio_array, sr_rate = sf.read(io.BytesIO(response.content), dtype="float32")
        _barge_in_detected.clear()
        _is_speaking.set()
        sd.play(audio_array, sr_rate)
        sd.wait()  # kehrt zurück wenn Wiedergabe endet oder sd.stop() aufgerufen wird
    except httpx.HTTPStatusError as e:
        log.error("TTS-Fehler (HTTP %d): %s", e.response.status_code, e)
    except Exception as e:
        log.error("TTS-Fehler: %s", e)
    finally:
        _is_speaking.clear()
        if not _barge_in_detected.is_set():
            # Normale Beendigung: TTS-Echo aus Queue verwerfen
            import time as _t; _t.sleep(0.4)
            while not _audio_queue.empty():
                try:
                    _audio_queue.get_nowait()
                except queue.Empty:
                    break
        emit("status", state="idle")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voxtral Voice Agent")
    parser.add_argument(
        "--backend",
        choices=["ollama", "claude"],
        default="ollama",
        help="LLM-Backend: 'ollama' (Standard) oder 'claude' (Claude Code CLI)",
    )
    args = parser.parse_args()

    ui_thread = threading.Thread(target=_start_ui_server, daemon=True, name="ui-server")
    ui_thread.start()
    log.info("Web-UI verfügbar unter: http://localhost:%d", UI_PORT)

    log.info("=== Voxtral-Agent gestartet (Backend: %s) ===", args.backend)

    if args.backend == "claude":
        claude_backend = ClaudeBackend()
        def get_response(user_text: str, messages: list) -> str:
            return claude_backend.chat(user_text)
    else:
        def get_response(user_text: str, messages: list) -> str:
            return get_llm_response(user_text, messages)

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

            bg_thread = threading.Thread(
                target=_bg_listener_loop, args=(recognizer, source),
                daemon=True, name="bg-listener",
            )
            bg_thread.start()

            while True:
                user_text = listen_and_transcribe()
                if not user_text:
                    continue
                if "beenden" in user_text.lower():
                    speak("Ich beende mich nun. Auf Wiedersehen!")
                    break

                ai_response = get_response(user_text, messages)
                speak(ai_response)
    except KeyboardInterrupt:
        log.info("Agent durch Benutzer beendet.")
    finally:
        _http.close()
