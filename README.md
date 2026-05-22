# Local Voice Agent – Projektdokumentation

> Lokaler, vollständig offline-fähiger Sprachassistent mit CUDA-Beschleunigung,  
> Ollama-LLM, Faster-Whisper STT, Voxtral TTS und erweiterbarem Tool-System.

---

## Inhaltsverzeichnis

1. [Übersicht](#übersicht)
2. [Architektur](#architektur)
3. [Komponenten & Technologien](#komponenten--technologien)
4. [Tool-System](#tool-system)
5. [Konfiguration](#konfiguration)
6. [Setup & Start](#setup--start)
7. [Git-Historie](#git-historie)
8. [Sicherheitshinweise](#sicherheitshinweise)
9. [Neues Tool hinzufügen](#neues-tool-hinzufügen)
10. [Bekannte Einschränkungen](#bekannte-einschränkungen)
5. [Paketmanagement](#paketmanagement)
6. [Konfiguration](#konfiguration)
7. [Setup & Start](#setup--start)
8. [Git-Historie](#git-historie)
9. [Sicherheitshinweise](#sicherheitshinweise)
10. [Neues Tool hinzufügen](#neues-tool-hinzufügen)
11. [Bekannte Einschränkungen](#bekannte-einschränkungen)

---

## Übersicht

| Eigenschaft | Wert |
|---|---------|
| Datei | `agent.py` (446 Zeilen) |
| Python | 3.12.13 (venv) |
| Paketmanager | [uv](https://github.com/astral-sh/uv) |
| LLM-Backend | Ollama lokal (`qwen2.5:7b`) |
| STT | Faster-Whisper `large-v3` (CUDA, int8_float16) |
| TTS | Voxtral-4B-TTS-2603 via HTTP (`localhost:8000`) |
| Sprache | Deutsch (`de`) |
| Tools | 8 registrierte Tools |

---

## Architektur

```
┌─────────────────────────────────────────────────────────┐
│                      agent.py                           │
│                                                         │
│  Mikrofon ──► STT (Faster-Whisper) ──► Text            │
│                                          │              │
│                                    get_llm_response()   │
│                                          │              │
│                               ┌──────────▼──────────┐  │
│                               │   Ollama (qwen2.5)  │  │
│                               │   Tool-Call-Schleife │  │
│                               └──────────┬──────────┘  │
│                                          │              │
│                              Tool-Dispatch via Registry │
│                             ┌────────────┴──────────┐   │
│                             │  TOOL_REGISTRY (dict) │   │
│                             └────────────┬──────────┘   │
│                                          │              │
│                      Text ◄─────────────┘              │
│                        │                               │
│               speak() ──► TTS (Voxtral) ──► Lautsprecher│
└─────────────────────────────────────────────────────────┘
```

### Gesprächsschleife (Hauptloop)

```
1. Mikrofon kalibrieren (2 Sek. Umgebungsgeräusche)
2. loop:
   a. Audio aufnehmen (timeout=5s, max=15s/Satz)
   b. Transkribieren (In-Memory, kein temp-File)
   c. LLM anfragen → ggf. Tools aufrufen (max. 5 Iterationen)
   d. Antwort vorlesen (Voxtral TTS)
   e. "beenden" → Abbruch
```

---

## Komponenten & Technologien

### STT – Speech-to-Text
- **Faster-Whisper** `large-v3`
- Gerät: `cuda`, Compute-Type: `int8_float16`
- In-Memory-Verarbeitung (kein `temp.wav` auf Platte)
- VAD-Filter aktiviert (filtert Stille heraus)
- `beam_size=5`, Sprache: `de`

### LLM – Sprachmodell
- **Ollama** mit Modell `qwen2.5:7b`
- Läuft vollständig lokal (`127.0.0.1:11434`)
- Tool-Calling via Ollama-nativer API
- Konversationshistorie wird mitgeführt und auf **20 Nachrichten** begrenzt

### TTS – Text-to-Speech
- **Mistral Voxtral-4B-TTS-2603**
- Lokaler HTTP-Server auf `localhost:8000`
- Format: WAV, Stimme: `de_female`
- Timeout: 120 Sekunden (für längere Texte)
- Wiedergabe via `sounddevice` + `soundfile`

### Abhängigkeiten

| Paket | Zweck |
|---|---|
| `faster-whisper` | STT |
| `ollama` | LLM-Client |
| `httpx` | HTTP-Requests (TTS, Suche, Wetter) |
| `sounddevice` | Audio-Wiedergabe |
| `soundfile` | WAV-Dekodierung |
| `SpeechRecognition` | Mikrofon-Aufnahme |
| `python-dotenv` | `.env`-Datei laden |
| `sympy` | Sichere Mathe-Auswertung |
| `psutil` | System-Metriken |

---

## Tool-System

### Decorator-Pattern

Tools werden über den `@tool`-Decorator registriert. Das JSON-Schema für Ollama wird **automatisch** aus den Python-Typhinweisen generiert – kein manuelles Schema-Pflegen nötig.

```python
@tool("Beschreibung des Tools", {"param": "Beschreibung des Parameters"})
def mein_tool(param: str) -> str:
    ...
```

Der Decorator erledigt automatisch:
- ✅ JSON-Schema-Generierung aus Typ-Hints (`str` → `"string"`, `float` → `"number"` etc.)
- ✅ `required`-Felder: Parameter ohne Default → required, mit Default → optional
- ✅ Eintrag in `TOOL_REGISTRY` für den Dispatch
- ✅ Eintrag in `tools`-Liste für Ollama

### Registrierte Tools

| Tool | Parameter | API-Key? | Beschreibung |
|---|---|:---:|---|
| `web_search` | `query: str` | ✅ Serper | Google-Suche via Serper API |
| `read_local_file` | `filepath: str` | ❌ | Textdatei aus `~/Documents` lesen |
| `write_file` | `filepath, content, append=False` | ❌ | Datei in `~/Documents` schreiben/anhängen |
| `list_files` | `subfolder=""` | ❌ | Inhalt von `~/Documents` auflisten |
| `get_datetime` | *(keine)* | ❌ | Aktuelles Datum & Uhrzeit |
| `calculate` | `expression: str` | ❌ | Mathe-Ausdruck auswerten (inkl. `15% von 340`) |
| `get_weather` | `location: str` | ❌ | Wetter via wttr.in (kostenlos) |
| `get_system_info` | *(keine)* | ❌ | CPU / RAM / Festplatte |
| `wiki_search` | `query: str` | ❌ | Wikipedia-Zusammenfassung (de) |
| `translate_text` | `text, target_lang, source_lang="de"` | ❌ | Übersetzung via MyMemory (kostenlos) |
| `get_news` | `category=""` | ❌ | Tagesschau RSS (Top/Inland/Ausland/Sport) |
| `quick_note` | `note: str` | ❌ | Notiz mit Zeitstempel → `~/Documents/notizen.md` |
| `open_application` | `app: str` | ❌ | App starten (Whitelist-gesichert) |

### Sicherheit bei Datei-Tools

Alle Datei-Tools verwenden **Path-Traversal-Schutz**:
```python
if not path.startswith(base_path + os.sep):
    return "Zugriff verweigert"
```
Der Zugriff ist auf `~/Documents` beschränkt. Ein `../../etc/passwd`-Angriff wird abgeblockt.

### Tool-Call-Schleife

```
MAX_TOOL_ITERATIONS = 5
```
Das LLM kann maximal **5 Tool-Aufrufe** pro Anfrage machen. Danach gibt es eine Standardantwort zurück. Verhindert Endlosschleifen.

---

## Konfiguration

### `agent.py` – Konstanten am Dateianfang

```python
DEVICE              = "cuda"           # oder "cpu"
VOXTRAL_URL         = "http://localhost:8000/v1/audio/speech"
MODEL_NAME          = "qwen2.5:7b"    # Ollama-Modellname

STT_LANGUAGE        = "de"
STT_BEAM_SIZE       = 5
LISTEN_TIMEOUT      = 5               # Sekunden warten bis Sprache kommt
PHRASE_TIME_LIMIT   = 15              # Max. Satzlänge in Sekunden

MAX_TOOL_ITERATIONS = 5               # Max. Tool-Calls pro Anfrage
MAX_HISTORY_MESSAGES = 20             # Max. Nachrichten in Kontext-History
```

### `.env` – Secrets (nicht in Git!)

```env
SERPER_API_KEY=dein-key-hier
```

> [!IMPORTANT]
> Die `.env`-Datei ist in `.gitignore` eingetragen und wird **niemals** in Git eingecheckt.
> API-Keys dürfen **nicht** in `agent.py` hardcoded werden.

Alternativer Weg (Terminal):
```bash
export SERPER_API_KEY="dein-key"
```
`load_dotenv()` überschreibt bereits gesetzte Env-Vars **nicht**.

---

## Setup & Start

### Voraussetzungen
- NVIDIA GPU mit CUDA
- [uv](https://github.com/astral-sh/uv) installiert
- Ollama läuft lokal mit `qwen2.5:7b` geladen
- Voxtral TTS-Server läuft auf Port 8000
- Mikrofon angeschlossen

### venv mit uv

Dieses Projekt verwendet **uv** als Paketmanager. Das venv liegt direkt im Projektordner.

> [!NOTE]
> `pip` ist in diesem venv **nicht** verfügbar. Alle Paketoperationen laufen über `uv`.

```bash
# Pakete installieren
uv pip install <paketname>

# Alle Abhängigkeiten auf einmal (falls requirements.txt vorhanden):
uv pip install -r requirements.txt

# Einzelnes Paket im Projektkontext:
uv add <paketname>
```

### Starten

```bash
# Im Projektverzeichnis:
source ./venv/bin/activate   # bash/zsh
# oder:
fish start_agent.fish         # fish shell

python agent.py
```

### Neues Paket hinzufügen

```bash
# Korrekt (uv):
uv pip install wikipedia-api

# NICHT:
# pip install wikipedia-api  ← pip nicht verfügbar
# ./venv/bin/pip install ... ← existiert nicht
```

### Beenden

- Sprache: *„beenden"* sagen
- Keyboard: `Ctrl+C`

---

## Git-Historie

| Commit | Beschreibung |
|---|---|
| `2888bb5` | Initial working state |
| `e5d5a5d` | Optimierter Agent (Logging, Tool-Registry, History-Trim, Security-Fix) |
| `1c06998` | Decorator-basiertes Tool-System (`@tool`) |
| `85faced` | Fix: SERPER_API_KEY via `.env` (python-dotenv) |
| `ea3fa89` | 6 neue Tools (datetime, calculate, weather, list/write_file, system_info) |

---

## Sicherheitshinweise

> [!WARNING]
> **API-Keys niemals in den Quellcode schreiben!**  
> Nutze ausschließlich die `.env`-Datei oder Umgebungsvariablen.

> [!CAUTION]
> **`run_shell_command` ist nicht implementiert** – bewusste Entscheidung.  
> Das Ausführen beliebiger Shell-Befehle durch ein LLM ist ohne strikte Whitelist zu gefährlich.

| Bereich | Maßnahme |
|---|---|
| API-Keys | `.env`-Datei, in `.gitignore` |
| Dateizugriff | Path-Traversal-Schutz, nur `~/Documents` |
| Tool-Loops | `MAX_TOOL_ITERATIONS = 5` |
| Kontext | `MAX_HISTORY_MESSAGES = 20` |
| Mathe | `sympy.sympify()` statt `eval()` |

---

## Neues Tool hinzufügen

So einfach ist es:

```python
# In agent.py, im Abschnitt "# --- TOOLS ---"

@tool(
    "Kurze Beschreibung was das Tool macht",
    {
        "param1": "Beschreibung von Parameter 1",
        "param2": "Beschreibung von Parameter 2",
    },
)
def mein_neues_tool(param1: str, param2: int = 0) -> str:
    """Längere Beschreibung für Entwickler."""
    # Implementierung hier
    return "Ergebnis als String"
```

**Das war's.** Kein Eintrag in Registry, kein manuelles Schema – der `@tool`-Decorator übernimmt alles.

Unterstützte Typen: `str`, `int`, `float`, `bool`, `list`, `dict`

---

## Bekannte Einschränkungen

| Problem | Ursache | Workaround |
|---|---|---|
| Hohe Startzeit | Faster-Whisper `large-v3` lädt ~3GB | Einmal starten, offen lassen |
| TTS-Latenz | Voxtral generiert serverseitig | Kürzere Antworten im System-Prompt |
| Kein Satzspeicher | History wird nach 20 Msgs gekürzt | `MAX_HISTORY_MESSAGES` erhöhen |
| Nur Deutsch | `STT_LANGUAGE = "de"` | Konstante ändern |
| Dateizugriff nur `~/Documents` | Sicherheits-Design | `base_path` in der Funktion anpassen |
