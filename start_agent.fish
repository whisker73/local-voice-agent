#!/usr/bin/env fish


echo "--- GPU Reinigungs-Check ---"
# Killt gezielt die vLLM Subprozesse, bevor das Skript überhaupt startet
set ZOMBIES (ps aux | grep -i "VLLM" | grep -v grep | awk '{print $2}')
if test -n "$ZOMBIES"
    echo "Found zombie VLLM processes: $ZOMBIES. Killing them..."
    kill -9 $ZOMBIES
end

echo "GPU ist sauber. Starte Systeme..."
# Projekt-Konfiguration
set PROJECT_DIR /home/whisker/local-voice-agent
set VENV_PATH $PROJECT_DIR/venv/bin/activate.fish

echo "--- Starte Voxtral & Agent System ---"

# 1. In das Verzeichnis wechseln
cd $PROJECT_DIR

# 2. Venv aktivieren
source $VENV_PATH

# 3. Voxtral Server starten (Hintergrund)
echo "Starte Voxtral Server mit vllm..."
vllm serve mistralai/Voxtral-4B-TTS-2603 --omni --gpu-memory-utilization 0.40 &
set VLLM_PID $last_pid

# 4. Warten bis der Server bereit ist (15s Puffer)
echo "Warte auf Initialisierung der GPU-Kerne..."
sleep 30

# 5. Agent starten (Vordergrund)
echo "Starte Agent.py..."
python agent.py

# 6. Aufräumen beim Beenden
echo "Beende Voxtral Server..."
kill $VLLM_PID
echo "Fertig."
