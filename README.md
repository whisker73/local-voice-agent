# Local Voice Agent – Projektdokumentation

> Lokaler, vollständig offline-fähiger Sprachassistent mit CUDA-Beschleunigung,  
> Ollama-LLM, Faster-Whisper STT, Voxtral TTS und erweiterbarem Tool-System.

---

## Inhaltsverzeichnis

1. [Übersicht](#übersicht)
2. [Architektur](#architektur)
3. [Komponenten & Technologien](#komponenten--technologien)
4. [Tool-System](#tool-system)
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
| Datei | `agent.py` |
| Python | 3.12.13 (venv) |
| Paketmanager | [uv](https://github.com/astral-sh/uv) |
| LLM-Backend | Ollama lokal (`qwen2.5:7b`) |
| STT | Faster-Whisper `large-v3` (CUDA, `float16`) |
| TTS | Voxtral-4B-TTS-2603 via HTTP (`localhost:8000`) |
| Sprache | Deutsch (`de`) |
| Tools | 15 registrierte Tools |

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
│                               │  Ollama (qwen2.5:7b)  │  │
│                               │  Tool-Call-Schleife  │  │
│                               └──────────┬──────────┘  │
│                                          │              │
│                             Tool-Dispatch via Registry  │
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
- Gerät: `cuda`, Compute-Type: `float16` (optimiert für RTX 3090 / Ampere)
- In-Memory-Verarbeitung (kein `temp.wav` auf Platte)
- VAD-Filter aktiviert (filtert Stille heraus)
- `beam_size=5`, Sprache: `de`

> [!NOTE]
> `float16` ist auf Ampere-GPUs (RTX 30xx) schneller als `int8_float16` und liefert
> bessere Transkriptionsqualität. Bei VRAM-Knappheit (< 8 GB) auf `int8_float16` wechseln.

### LLM – Sprachmodell
- **Ollama** mit Modell `qwen2.5:7b`
- Läuft vollständig lokal (`127.0.0.1:11434`)
- Tool-Calling via Ollama-nativer API
- Konversationshistorie wird mitgeführt und auf **20 Nachrichten** begrenzt
- Zuverlässiges natives Tool-Calling (kein Fallback-Parser nötig)
- **Fallback-Parser** `_parse_text_tool_calls()` bleibt als Absicherung erhalten, falls das Modell ausnahmsweise Klartext statt strukturiertem API-Call ausgibt
- **Parameter-Filterung**: Unbekannte Argumente (z.B. modellspezifische Extras wie `toolbench`) werden vor dem Tool-Aufruf automatisch entfernt

### TTS – Text-to-Speech
- **Mistral Voxtral-4B-TTS-2603**
- Lokaler HTTP-Server auf `localhost:8000`
- Format: WAV, Stimme: `de_female`
- Timeout: 120 Sekunden (für längere Texte)
- Wiedergabe via `sounddevice` + `soundfile`

### HTTP-Client
Alle externen Requests (TTS, Suche, Wetter, Wikipedia, …) laufen über einen einzigen
`httpx.Client`-Instanz (`_http`) auf Modulebene. Das ermöglicht **Connection-Pooling** –
kein neuer TCP-Handshake pro Anfrage. Der Client wird beim Beenden sauber geschlossen.

### Abhängigkeiten

| Paket | Zweck |
|---|---|
| `faster-whisper` | STT |
| `ollama` | LLM-Client |
| `httpx` | HTTP-Requests mit Connection-Pooling |
| `sounddevice` | Audio-Wiedergabe |
| `soundfile` | WAV-Dekodierung |
| `SpeechRecognition` | Mikrofon-Aufnahme |
| `python-dotenv` | `.env`-Datei laden |
| `sympy` | Sichere Mathe-Auswertung |
| `psutil` | System-Metriken |
| `urllib.parse` | URL-Encoding (Stdlib, kein Install nötig) |
| `json` | JSON-Parsing für HA-Dienste & Fallback-Parser (Stdlib) |

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
| `get_news` | `category=""` | ❌ | Tagesschau RSS – Top / Inland / Ausland / Wirtschaft / Sport |
| `quick_note` | `note: str` | ❌ | Notiz mit Zeitstempel → `~/Documents/notizen.md` |
| `open_application` | `app: str` | ❌ | App starten (Whitelist-gesichert) |
| `ha_get_state` | `entity_id=""` | ✅ HA-Token | Zustand eines HA-Geräts abfragen (oder alle auflisten) |
| `ha_call_service` | `domain, service, entity_id, extra=""` | ✅ HA-Token | HA-Dienst aufrufen (Licht, Heizung, Szenen …) |

### Sicherheit bei Datei-Tools

Alle Datei-Tools verwenden **Path-Traversal-Schutz** mit explizitem `os.sep`-Check:
```python
# Verhindert auch Zugriff auf Pfade mit gleichem Präfix (z.B. ~/Documents2)
if not (path == base_path or path.startswith(base_path + os.sep)):
    return "Zugriff verweigert"
```
Der Zugriff ist auf `~/Documents` beschränkt. Ein `../../etc/passwd`-Angriff wird abgeblockt.

### Tool-Call-Schleife

```
MAX_TOOL_ITERATIONS = 5
```
Das LLM kann maximal **5 Tool-Aufrufe** pro Anfrage machen. Danach gibt es eine Standardantwort zurück. Verhindert Endlosschleifen.

### Fallback-Parser für Tool-Calls

`_parse_text_tool_calls()` greift als Absicherung, falls das Modell einen Tool-Call ausnahmsweise als Klartext statt als strukturierten API-Call ausgibt. Im Log sichtbar als:
```
[WARNING] Modell gab Tool-Call als Klartext aus – Fallback-Parser aktiv
```

### Parameter-Filterung

Manche Modelle senden nicht im Schema definierte Zusatzparameter (z.B. `toolbench`). Diese werden vor jedem Tool-Aufruf anhand der Funktionssignatur herausgefiltert, damit kein `TypeError` auftritt.

### Home Assistant

Zwei Tools steuern Home Assistant per Sprache:

| Tool | Wofür |
|---|---|
| `ha_get_state` | Zustand abfragen: Temperatur, ob Licht an, Akkustand … |
| `ha_call_service` | Steuern: Licht, Heizung, Schalter, Szenen, Automationen … |

**Beispiel-Sprachbefehle:**
- *„Mach das Licht im Wohnzimmer an"* → `ha_call_service(domain="light", service="turn_on", entity_id="light.wohnzimmer")`
- *„Wie warm ist es im Schlafzimmer?"* → `ha_get_state(entity_id="sensor.schlafzimmer_temperatur")`
- *„Stell die Heizung auf 21 Grad"* → `ha_call_service(..., extra='{"temperature": 21}')`
- *„Welche Geräte habe ich?"* → `ha_get_state()` ohne Parameter → listet alle Entitäten auf

---

## Paketmanagement

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

# Home Assistant (Long-Lived Access Token aus HA → Profil → Sicherheit)
HA_URL=http://homeassistant.local:8123
HA_TOKEN=dein-token-hier
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
- NVIDIA GPU mit CUDA (empfohlen: RTX 3090 oder besser für `float16`-Betrieb)
- [uv](https://github.com/astral-sh/uv) installiert
- Ollama läuft lokal mit `qwen2.5:7b` geladen
- Mikrofon angeschlossen
- Home Assistant erreichbar (optional, für HA-Tools)

### Starten

**Empfohlen – `start_agent.fish`** (fish shell):
```bash
fish start_agent.fish
```
Das Script erledigt automatisch:
1. Ollama-Modell aus VRAM entladen (gibt Speicher für vLLM frei)
2. Alte vLLM-Prozesse beenden
3. Voxtral TTS-Server starten (Log: `/tmp/voxtral_server.log`)
4. Per Health-Check warten bis der Server bereit ist (max. 180 s)
5. `agent.py` starten
6. Voxtral beim Beenden sauber stoppen
7. Ollama-Modell aus VRAM entladen (gibt Speicher frei)

**Manuell** (zwei Terminals):
```bash
# Terminal 1 – Voxtral TTS-Server:
vllm serve mistralai/Voxtral-4B-TTS-2603 --omni --gpu-memory-utilization 0.40

# Terminal 2 – Agent (nach vollständigem Serverstart):
source ./venv/bin/activate
python agent.py
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
| `da73b63` | 5 neue Tools (wiki, translate, news, quick_note, open_app) |
| `5d16983` | Refactor: Connection-Pooling, `float16`-STT, Import-Cleanup, Bugfixes |
| `95e66d5` | fix start_agent.fish: Health-Check, vLLM-Log, Crash-Erkennung |
| `45a09f9` | fix start_agent.fish: Ollama-VRAM vor vLLM-Start freigeben |
| `d0f3ad4` | feat: Home Assistant Integration (`ha_get_state`, `ha_call_service`) |
| `46002d5` | fix: Fallback-Parser für Text-Tool-Calls |
| `4aea456` | docs: README aktualisiert – HA-Integration, Fallback-Parser, start_agent.fish |
| *(aktuell)* | feat: qwen2.5:7b; HA-Entityliste gruppiert mit IDs; Parameter-Filterung; Ollama-Entladung beim Stop |

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
| Dateizugriff | Path-Traversal-Schutz mit `os.sep`, nur `~/Documents` |
| URL-Injection | `urllib.parse.quote()` bei URLs mit Nutzereingaben (z.B. `get_weather`) |
| Tool-Loops | `MAX_TOOL_ITERATIONS = 5` |
| Kontext | `MAX_HISTORY_MESSAGES = 20` |
| Mathe | `sympy.sympify()` statt `eval()` |
| App-Start | Whitelist `_ALLOWED_APPS`, kein Shell-Injection (Liste statt String) |

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
| Hohe Startzeit | Faster-Whisper `large-v3` lädt ~3 GB | Einmal starten, offen lassen |
| TTS-Latenz | Voxtral generiert serverseitig | Kürzere Antworten im System-Prompt |
| Kein Satzspeicher | History wird nach 20 Msgs gekürzt | `MAX_HISTORY_MESSAGES` erhöhen |
| Nur Deutsch | `STT_LANGUAGE = "de"` | Konstante ändern |
| Dateizugriff nur `~/Documents` | Sicherheits-Design | `base_path` in der Funktion anpassen |
| `sympy` langsamer erster Start | Lazy-Import beim ersten `calculate`-Aufruf | Unvermeidbar, ~1–2 s Verzögerung |
| Tool-Calls als Klartext | Modell gibt gelegentlich Klartext aus | Fallback-Parser greift automatisch |
| Unbekannte Tool-Parameter | Modell sendet extra Felder (z.B. `toolbench`) | Werden automatisch vor dem Aufruf gefiltert |
| HA-Entity-ID unbekannt | IDs müssen exakt stimmen | *„Welche Geräte habe ich?"* → alle auflisten lassen (inkl. Entity-IDs) |
