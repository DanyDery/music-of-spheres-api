"""
Music of the Spheres — FastAPI Backend
═══════════════════════════════════════
Endpoints:
  POST /generate            → natal chart + WAV + PNG
  GET  /audio/{id}          → full WAV mix
  GET  /image/{id}          → PNG natal map
  POST /generate/planets    → 10 individual planet WAVs
  GET  /planet-audio/{id}/{planet} → single planet WAV
  GET  /health              → server status

Install:
  pip install fastapi uvicorn kerykeion numpy pillow
              geopy timezonefinder certifi

Run:
  uvicorn main:app --reload --port 8000
"""

import ssl, certifi, os, uuid
os.environ['SSL_CERT_FILE']      = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from datetime import date
import numpy as np
import wave as wv
import struct, math, sys

sys.path.insert(0, str(Path(__file__).parent))
import music_of_spheres as m

app = FastAPI(
    title="Music of the Spheres API",
    description="Pythagorean planetary matrix generator • Kusto 1978 • 432 Hz",
    version="1.0.0",
)

# CORS — allow frontend to call the API
# In production replace "*" with your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("/tmp/spheres")
OUTPUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════
# REQUEST SCHEMAS
# ══════════════════════════════════════════════

class GenerateRequest(BaseModel):
    year:     int = Field(..., ge=1900, le=2100, example=1990)
    month:    int = Field(..., ge=1,    le=12,   example=3)
    day:      int = Field(..., ge=1,    le=31,   example=15)
    hour:     int = Field(12,  ge=0,    le=23,   example=14)
    minute:   int = Field(0,   ge=0,    le=59,   example=30)
    city:     str = Field(..., min_length=2,     example="Kyiv")
    duration: int = Field(60,  ge=10,   le=600,  example=60)


# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════

def build_session(req: GenerateRequest):
    """Create session directory and build natal chart."""
    try:
        birthdate = date(req.year, req.month, req.day)
    except ValueError as e:
        raise HTTPException(400, f"Invalid date: {e}")
    try:
        chart = m.build_chart(
            req.year, req.month, req.day,
            req.hour, req.minute, req.city
        )
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Geocoding error: {e}")

    session_id  = str(uuid.uuid4())[:8]
    session_dir = OUTPUT_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    return session_id, chart, birthdate, session_dir


def make_planet_wav(planet: str, d: dict,
                    duration: int, session_dir: Path) -> Path:
    """
    Generate a solo WAV for a single planet.
    Includes all three Pythagorean intervals:
      perfect fifth (×1.5), perfect fourth (×1.333), octave (×2.0)
    """
    freq    = d["freq"]
    abs_deg = d["abs_deg"]
    beat_hz = d["beat_hz"]
    phase   = (abs_deg / 360.0) * 2 * math.pi
    amp     = 0.55  # full amplitude for solo listening

    t    = np.linspace(0, duration,
                       int(m.SAMPLE_RATE * duration), endpoint=False)

    # Planetary pulse from orbital period
    lfo  = 0.65 + 0.35 * np.sin(2 * math.pi * beat_hz * t)

    # String timbre (sawtooth harmonics) — Pythagoras's monochord
    tone = amp * lfo * m.string_tone(freq,        t, phase)
    q5   = amp * 0.35 * lfo * m.string_tone(freq * 1.500, t, phase)  # perfect fifth
    q4   = amp * 0.25 * lfo * m.string_tone(freq * 1.333, t, phase)  # perfect fourth
    oct_ = amp * 0.15 * lfo * m.string_tone(freq * 2.000, t, phase)  # octave
    sig  = tone + q5 + q4 + oct_

    sig  = m.envelope(sig, attack=2.0, release=3.0)
    sig  = m.reverb(sig, decay=0.3, delay_ms=80, num_echoes=4)
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig = sig / peak * 0.85

    # Stereo panning from ecliptic position
    pan   = 0.5 + 0.45 * math.sin(math.radians(abs_deg))
    left  = np.clip(sig * (1 - pan), -1, 1)
    right = np.clip(sig * pan,       -1, 1)

    fpath = session_dir / f"planet_{planet.lower()}.wav"
    with wv.open(str(fpath), 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(m.SAMPLE_RATE)
        for l, r in zip(left, right):
            wf.writeframes(struct.pack('<hh',
                int(l * 32767), int(r * 32767)))
    return fpath


# ══════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "service": "Music of the Spheres API v1.0"}


@app.post("/generate")
def generate(req: GenerateRequest):
    """
    Main endpoint.
    Builds natal chart via Swiss Ephemeris, generates full
    planetary WAV mix and PNG natal map.
    Returns chart data + download links.
    """
    session_id, chart, birthdate, session_dir = build_session(req)

    orig = Path.cwd()
    os.chdir(session_dir)
    try:
        m.generate_audio(chart, birthdate, req.duration)
        m.generate_image(chart, birthdate)
    finally:
        os.chdir(orig)

    return {
        "session_id": session_id,
        "birthdate":  birthdate.strftime("%d.%m.%Y"),
        "city":       req.city,
        "tuning":     "432 Hz",
        "theory":     "Pythagorean • Kusto 1978",
        "audio_url":  f"/audio/{session_id}",
        "image_url":  f"/image/{session_id}",
        "planets": {
            name: {
                "freq":       d["freq"],
                "base_freq":  d["base_freq"],
                "bpm":        round(d["bpm"], 4),
                "strength":   round(d["strength"], 3),
                "weight":     d["weight"],
                "sign":       d["sign"],
                "sign_short": d["sign_short"],
                "deg":        round(d["deg"], 2),
                "abs_deg":    round(d["abs_deg"], 2),
                "color":      list(d["color"]),
            }
            for name, d in chart.items()
        }
    }


@app.post("/generate/planets")
def generate_planets(req: GenerateRequest):
    """
    Generate a separate WAV for each of the 10 planets.
    Used by the frontend for interactive planet on/off mixing
    via Web Audio API.
    """
    session_id, chart, birthdate, session_dir = build_session(req)

    planet_urls = {}
    for planet in m.PLANET_ORDER:
        if planet not in chart:
            continue
        make_planet_wav(planet, chart[planet], req.duration, session_dir)
        planet_urls[planet] = f"/planet-audio/{session_id}/{planet.lower()}"

    return {
        "session_id":  session_id,
        "planet_urls": planet_urls,
        "chart": {
            name: {
                "freq":   d["freq"],
                "bpm":    round(d["bpm"], 4),
                "sign":   d["sign"],
                "color":  list(d["color"]),
                "weight": d["weight"],
            }
            for name, d in chart.items()
        }
    }


@app.get("/audio/{session_id}")
def get_audio(session_id: str):
    """Download the full planetary WAV mix."""
    files = list((OUTPUT_DIR / session_id).glob("spheres_*.wav"))
    if not files:
        raise HTTPException(404, "Audio not found")
    return FileResponse(files[0], media_type="audio/wav",
                        filename=files[0].name)


@app.get("/image/{session_id}")
def get_image(session_id: str):
    """Download the PNG natal map."""
    files = list((OUTPUT_DIR / session_id).glob("spheres_*.png"))
    if not files:
        raise HTTPException(404, "Image not found")
    return FileResponse(files[0], media_type="image/png",
                        filename=files[0].name)


@app.get("/planet-audio/{session_id}/{planet_name}")
def get_planet_audio(session_id: str, planet_name: str):
    """Download WAV for a single planet (sun, moon, mars, ...)."""
    fpath = OUTPUT_DIR / session_id / f"planet_{planet_name}.wav"
    if not fpath.exists():
        raise HTTPException(404, f"Planet '{planet_name}' not found")
    return FileResponse(fpath, media_type="audio/wav")
