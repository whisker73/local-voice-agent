import httpx

print("Lese Audiodatei...")
with open("referenz.wav", "rb") as f:
    # API verlangt 'audio_sample' statt 'audio' und 'consent'
    files = {"audio_sample": ("referenz.wav", f, "audio/wav")}
    data = {
        "name": "holger_voice",
        "consent": "true"
    }

    print("Sende Stimme an den Server...")
    try:
        response = httpx.post("http://localhost:8000/v1/audio/voices", data=data, files=files)
        response.raise_for_status()
        print("Erfolg! Deine Stimme wurde geklont und im Server registriert.")
    except httpx.HTTPStatusError as e:
        print(f"Fehler {e.response.status_code}")
        print(f"Server sagt: {e.response.text}")
