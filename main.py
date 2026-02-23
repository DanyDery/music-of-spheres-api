"""
Music of the Spheres — FastAPI Backend v1.2
═══════════════════════════════════════════
Architecture:
  POST /generate  — generates EVERYTHING in parallel:
    - full mix WAV
    - natal map PNG
    - 10 individual planet WAVs
  All three run simultaneously in ThreadPoolExecutor.
  Frontend fetches planet files individually after — instant.

  /generate/planets is now deprecated (kept for compatibility).

Endpoints:
  POST /generate                   → all files at once
  GET  /audio/{id}                 → full WAV mix
  GET  /image/{id}                 → PNG natal map
  GET  /planet-audio/{id}/{planet} → single planet WAV (pre-generated)
  GET  /health                     → server status
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
from concurrent.futures import ThreadPoolExecutor
import asyncio
import numpy as np
import wave as wv
import struct, math, sys

sys.path.insert(0, str(Path(__file__).parent))
import music_of_spheres as m

app = FastAPI(
    title="Music of the Spheres API",
    description="Pythagorean planetary matrix generator • Kusto 1978 • 432 Hz",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import Request
from fastapi.responses import Response

@app.middleware("http")
async def cors_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        return Response(status_code=200, headers={
            "Access-Control-Allow-Origin":  "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        })
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

OUTPUT_DIR = Path("/tmp/spheres")
OUTPUT_DIR.mkdir(exist_ok=True)

# 4 workers: audio + image + up to 10 planets all at once
executor = ThreadPoolExecutor(max_workers=12)


# ══════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════

class GenerateRequest(BaseModel):
    year:     int = Field(..., ge=1900, le=2100, example=1990)
    month:    int = Field(..., ge=1,    le=12,   example=3)
    day:      int = Field(..., ge=1,    le=31,   example=15)
    hour:     int = Field(12,  ge=0,    le=23,   example=14)
    minute:   int = Field(0,   ge=0,    le=59,   example=30)
    city:     str = Field(..., min_length=2,     example="Kyiv")
    duration: int = Field(30,  ge=10,   le=600,  example=30)


# ══════════════════════════════════════════════
# WORKER FUNCTIONS (run in thread pool)
# ══════════════════════════════════════════════

def _gen_audio(chart, birthdate, duration, session_dir):
    orig = Path.cwd(); os.chdir(session_dir)
    try:    return m.generate_audio(chart, birthdate, duration)
    finally: os.chdir(orig)

def _gen_image(chart, birthdate, session_dir):
    orig = Path.cwd(); os.chdir(session_dir)
    try:    return m.generate_image(chart, birthdate)
    finally: os.chdir(orig)

def _gen_planet(planet, d, duration, session_dir):
    """Generate solo WAV for one planet — runs in thread pool."""
    freq    = d["freq"]
    abs_deg = d["abs_deg"]
    beat_hz = d["beat_hz"]
    phase   = (abs_deg / 360.0) * 2 * math.pi
    amp     = 0.55

    t    = np.linspace(0, duration, int(m.SAMPLE_RATE * duration), endpoint=False)
    lfo  = 0.65 + 0.35 * np.sin(2 * math.pi * beat_hz * t)
    tone = amp * lfo * m.string_tone(freq,        t, phase)
    q5   = amp * 0.35 * lfo * m.string_tone(freq * 1.500, t, phase)
    q4   = amp * 0.25 * lfo * m.string_tone(freq * 1.333, t, phase)
    oct_ = amp * 0.15 * lfo * m.string_tone(freq * 2.000, t, phase)
    sig  = m.envelope(tone + q5 + q4 + oct_, attack=1.5, release=2.0)
    sig  = m.reverb(sig, decay=0.3, delay_ms=80, num_echoes=4)
    peak = np.max(np.abs(sig))
    if peak > 0: sig = sig / peak * 0.85

    pan   = 0.5 + 0.45 * math.sin(math.radians(abs_deg))
    left  = np.clip(sig * (1 - pan), -1, 1)
    right = np.clip(sig * pan,       -1, 1)

    fpath = session_dir / f"planet_{planet.lower()}.wav"
    with wv.open(str(fpath), 'w') as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(m.SAMPLE_RATE)
        for l, r in zip(left, right):
            wf.writeframes(struct.pack('<hh', int(l * 32767), int(r * 32767)))
    return fpath


# ══════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "service": "Music of the Spheres API v1.2"}


@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Generates EVERYTHING in parallel:
      - full mix WAV
      - natal map PNG
      - 10 individual planet WAVs

    All run simultaneously → total time ≈ time of slowest single task.
    Planet files are pre-generated so frontend can fetch them instantly.
    """
    # Build chart
    try:
        birthdate = date(req.year, req.month, req.day)
    except ValueError as e:
        raise HTTPException(400, f"Invalid date: {e}")
    try:
        chart = m.build_chart(req.year, req.month, req.day,
                              req.hour, req.minute, req.city)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Geocoding error: {e}")

    session_id  = str(uuid.uuid4())[:8]
    session_dir = OUTPUT_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    loop = asyncio.get_event_loop()

    # Launch all tasks simultaneously
    audio_fut  = loop.run_in_executor(executor, _gen_audio, chart, birthdate, req.duration, session_dir)
    image_fut  = loop.run_in_executor(executor, _gen_image, chart, birthdate, session_dir)
    planet_futs = [
        loop.run_in_executor(executor, _gen_planet, planet, chart[planet], req.duration, session_dir)
        for planet in m.PLANET_ORDER if planet in chart
    ]

    # Wait for everything
    await asyncio.gather(audio_fut, image_fut, *planet_futs)

    return {
        "session_id": session_id,
        "birthdate":  birthdate.strftime("%d.%m.%Y"),
        "city":       req.city,
        "tuning":     "432 Hz",
        "theory":     "Pythagorean • Kusto 1978",
        "audio_url":  f"/audio/{session_id}",
        "image_url":  f"/image/{session_id}",
        # Planet URLs — files already exist, fetching is instant
        "planet_urls": {
            planet: f"/planet-audio/{session_id}/{planet.lower()}"
            for planet in m.PLANET_ORDER if planet in chart
        },
        "planets": {
            name: {
                "freq":      d["freq"],
                "base_freq": d["base_freq"],
                "bpm":       round(d["bpm"], 4),
                "strength":  round(d["strength"], 3),
                "weight":    d["weight"],
                "sign":      d["sign"],
                "sign_short":d["sign_short"],
                "deg":       round(d["deg"], 2),
                "abs_deg":   round(d["abs_deg"], 2),
                "color":     list(d["color"]),
            }
            for name, d in chart.items()
        }
    }


@app.post("/generate/planets")
async def generate_planets(req: GenerateRequest):
    """
    Deprecated — kept for compatibility.
    Now /generate already includes planet_urls.
    """
    raise HTTPException(410, "Deprecated. Use POST /generate — it now includes planet_urls.")


@app.get("/audio/{session_id}")
def get_audio(session_id: str):
    files = list((OUTPUT_DIR / session_id).glob("spheres_*.wav"))
    if not files: raise HTTPException(404, "Audio not found")
    return FileResponse(files[0], media_type="audio/wav", filename=files[0].name)


@app.get("/image/{session_id}")
def get_image(session_id: str):
    files = list((OUTPUT_DIR / session_id).glob("spheres_*.png"))
    if not files: raise HTTPException(404, "Image not found")
    return FileResponse(files[0], media_type="image/png", filename=files[0].name)


@app.get("/planet-audio/{session_id}/{planet_name}")
def get_planet_audio(session_id: str, planet_name: str):
    fpath = OUTPUT_DIR / session_id / f"planet_{planet_name}.wav"
    if not fpath.exists(): raise HTTPException(404, f"Planet '{planet_name}' not found")
    return FileResponse(fpath, media_type="audio/wav")
