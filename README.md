# 🌌 Music of the Spheres — API

Personalized planetary sound matrix generator based on Pythagorean theory of harmony.

## Theory

- **Pythagoras**: 12 zodiac signs = 12 chromatic semitones. Three sacred intervals: perfect fifth (3:2), perfect fourth (4:3), octave (2:1).
- **Hans Cousto (1978)**: planetary base frequencies derived from orbital periods via octave transposition.
- **Formula**: `freq = base × 2^(semitone/12) × 2^(degree/29/12)`
- **Tuning**: 432 Hz — natural tuning system used by Cousto.
- **Timbre**: sawtooth wave (6 harmonics) — historically grounded in Pythagoras's monochord experiments.
- **Rhythm**: planetary BPM derived from orbital periods using the same octave method as frequencies.
- **Earth drone**: foundation of the system (per Robert Fludd's Celestial Monochord — Terra at the base).
- **Binaural beat**: Sun ↔ Moon frequencies create a unique beat for each natal chart.

Every sound is mathematically connected to the exact moment and place of birth via Swiss Ephemeris.

## Endpoints

### `POST /generate`
Build natal chart and generate full planetary mix (WAV + PNG).

**Request:**
```json
{
  "year": 1990, "month": 3, "day": 15,
  "hour": 14, "minute": 30,
  "city": "Kyiv",
  "duration": 60
}
```

**Response:**
```json
{
  "session_id": "a1b2c3d4",
  "birthdate": "15.03.1990",
  "tuning": "432 Hz",
  "theory": "Pythagorean • Kusto 1978",
  "audio_url": "/audio/a1b2c3d4",
  "image_url": "/image/a1b2c3d4",
  "planets": {
    "Sun":  { "freq": 245.70, "bpm": 3.94, "sign": "Pis", "color": [255,200,50] },
    "Moon": { "freq": 315.45, "bpm": 52.71, "sign": "Sco", "color": [200,210,255] }
  }
}
```

### `POST /generate/planets`
Generate a separate WAV for each of the 10 planets.
Used by the frontend for interactive on/off planet mixing.

### `GET /audio/{session_id}` — download full WAV mix
### `GET /image/{session_id}` — download PNG natal map
### `GET /planet-audio/{session_id}/{planet}` — single planet WAV (sun, moon, mars...)
### `GET /health` — server status

## Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Interactive docs: **http://localhost:8000/docs**

## Deploy on Railway

1. Push this repo to GitHub
2. railway.app → New Project → Deploy from GitHub
3. Add start command:

```
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Stack

| Library | Purpose |
|---------|---------|
| FastAPI | API framework |
| kerykeion | Swiss Ephemeris natal chart |
| numpy | Audio synthesis |
| Pillow | Image generation |
| geopy + timezonefinder | City geocoding with cache |

## Planet Frequencies (432 Hz tuning)

| Planet  | Base Hz | Sign modulates frequency |
|---------|---------|--------------------------|
| Sun     | 126.22  | × semitone factor 0..11  |
| Moon    | 210.42  | fastest rhythm ~53 BPM   |
| Mercury | 141.27  | |
| Venus   | 221.23  | |
| Mars    | 144.72  | |
| Jupiter | 183.58  | |
| Saturn  | 147.85  | |
| Uranus  | 207.36  | |
| Neptune | 211.44  | |
| Pluto   | 140.25  | slowest rhythm ~0.5 BPM  |
