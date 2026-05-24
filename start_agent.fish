#!/usr/bin/env fish

set PROJECT_DIR  /home/whisker/local-voice-agent
set VENV_PATH    $PROJECT_DIR/venv/bin/activate.fish
set VLLM_LOG     /tmp/voxtral_server.log
set VLLM_PORT    8000
set HEALTH_URL   "http://localhost:$VLLM_PORT/health"
set OLLAMA_URL   "http://localhost:11434"
set MAX_WAIT     180  # Sekunden bis Timeout

# --- Ollama-Modell aus VRAM entladen ---
echo "--- Ollama VRAM freigeben ---"
if curl -sf "$OLLAMA_URL/api/tags" > /dev/null 2>&1
    echo "Ollama läuft – entlade Modelle aus VRAM..."
    # keep_alive=0 weist Ollama an, das Modell sofort aus dem Speicher zu entfernen
    curl -sf -X POST "$OLLAMA_URL/api/generate" \
        -d "{\"model\": \"qwen2.5:7b\", \"keep_alive\": 0}" > /dev/null 2>&1
    sleep 1
    echo "Ollama-VRAM freigegeben."
else
    echo "Ollama nicht erreichbar – überspringe."
end

# --- Alte vLLM-Prozesse aufräumen ---
echo "--- GPU-Reinigungs-Check ---"
set ZOMBIES (ps aux | grep -iE "vllm|voxtral" | grep -v grep | awk '{print $2}')
if test -n "$ZOMBIES"
    echo "Beende vorhandene vLLM-Prozesse: $ZOMBIES"
    kill -9 $ZOMBIES
    sleep 2
end
echo "GPU ist sauber."

# --- Venv aktivieren ---
cd $PROJECT_DIR
source $VENV_PATH

# --- Prüfen ob vllm verfügbar ist ---
if not type -q vllm
    echo "FEHLER: 'vllm' nicht gefunden. Im venv oder im PATH installiert?"
    exit 1
end

# --- Voxtral TTS-Server im Hintergrund starten ---
# Output geht in Logdatei – kein Durcheinander im Terminal
echo "Starte Voxtral TTS-Server (Log: $VLLM_LOG)..."
vllm serve mistralai/Voxtral-4B-TTS-2603 --omni --gpu-memory-utilization 0.40 > $VLLM_LOG 2>&1 &
set VLLM_PID $last_pid
echo "vLLM gestartet (PID $VLLM_PID)"

# --- Warten bis der Server wirklich bereit ist ---
echo "Warte auf Server-Bereitschaft (max. $MAX_WAIT Sekunden)..."
set elapsed 0
while test $elapsed -lt $MAX_WAIT
    # Prozess abgestürzt?
    if not kill -0 $VLLM_PID 2>/dev/null
        echo ""
        echo "FEHLER: vLLM-Prozess ist abgestürzt! Letzte Log-Zeilen:"
        tail -30 $VLLM_LOG
        exit 1
    end

    # Server antwortet?
    if curl -sf $HEALTH_URL > /dev/null 2>&1
        echo ""
        echo "Server ist bereit (nach $elapsed Sekunden)."
        break
    end

    printf "."
    sleep 3
    set elapsed (math $elapsed + 3)
end

if test $elapsed -ge $MAX_WAIT
    echo ""
    echo "FEHLER: Server nicht bereit nach $MAX_WAIT Sekunden. Letzte Log-Zeilen:"
    tail -30 $VLLM_LOG
    kill $VLLM_PID 2>/dev/null
    exit 1
end

# --- Agent starten (Vordergrund) ---
echo "Starte Agent..."
python agent.py

# --- Aufräumen ---
echo "Beende Voxtral Server (PID $VLLM_PID)..."
kill $VLLM_PID 2>/dev/null

# --- Ollama-Modell aus VRAM entladen ---
echo "--- Ollama VRAM freigeben ---"
if curl -sf "$OLLAMA_URL/api/tags" > /dev/null 2>&1
    curl -sf -X POST "$OLLAMA_URL/api/generate" \
        -d "{\"model\": \"qwen2.5:7b\", \"keep_alive\": 0}" > /dev/null 2>&1
    echo "Ollama-VRAM freigegeben."
else
    echo "Ollama nicht erreichbar – überspringe."
end

echo "Fertig."
