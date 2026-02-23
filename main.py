"""
Music of the Spheres — FastAPI Backend v1.3
═══════════════════════════════════════════
Architecture:
  POST /generate
    1. Builds natal chart
    2. Starts audio + image generation (awaited — ~20s)
    3. Starts planet WAV generation IN BACKGROUND (asyncio.create_task)
    4. Returns response immediately with session_id + planet_urls

  GET /planet-audio/{session_id}/{planet}
    - If file ready   → returns WAV instantly
    - If not ready yet → 202 Accepted (frontend retries after 2s)

  This way Railway never hits the 30s timeout on planet tracks.
  Each individual planet file request is < 1s once ready.
"""

import ssl, certifi, os, uuid, asyncio
os.environ['SSL_CERT_FILE']      = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from pathlib import Path
from datetime import date
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import wave as wv
import struct, math, sys

sys.path.insert(0, str(Path(__file__).parent))
import music_of_spheres as m

app = FastAPI(
    title="Music of the Spheres API",
    description="Pythagorean planetary matrix generator • Kusto 1978 • 432 Hz",
    version="1.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Thread pool for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=4)

PLANET_DURATION = 10  # seconds per planet track


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
# WORKER FUNCTIONS (CPU-bound, run in thread pool)
# ══════════════════════════════════════════════

def _gen_audio(chart, birthdate, duration, session_dir):
    orig = Path.cwd(); os.chdir(session_dir)
    try:    return m.generate_audio(chart, birthdate, duration)
    finally: os.chdir(orig)

def _gen_image(chart, birthdate, session_dir):
    orig = Path.cwd(); os.chdir(session_dir)
    try:    return m.generate_image(chart, birthdate)
    finally: os.chdir(orig)

def _gen_planet(planet: str, d: dict, duration: int, session_dir: Path) -> Path:
    """
    Generate solo WAV for one planet.
    Three Pythagorean intervals: fifth (3:2), fourth (4:3), octave (2:1).
    """
    freq    = d["freq"]
    abs_deg = d["abs_deg"]
    beat_hz = d["beat_hz"]
    phase   = (abs_deg / 360.0) * 2 * math.pi
    amp     = 0.55

    t    = np.linspace(0, duration, int(m.SAMPLE_RATE * duration), endpoint=False)
    lfo  = 0.65 + 0.35 * np.sin(2 * math.pi * beat_hz * t)
    tone = amp * lfo * m.string_tone(freq,        t, phase)
    q5   = amp * 0.35 * lfo * m.string_tone(freq * 1.500, t, phase)  # perfect fifth
    q4   = amp * 0.25 * lfo * m.string_tone(freq * 1.333, t, phase)  # perfect fourth
    oct_ = amp * 0.15 * lfo * m.string_tone(freq * 2.000, t, phase)  # octave
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


async def _gen_planets_background(chart: dict, session_dir: Path):
    """
    Generate all 10 planet WAVs in background after /generate returns.
    Each planet runs in thread pool — all 10 in parallel.
    """
    loop = asyncio.get_event_loop()
    futures = [
        loop.run_in_executor(executor, _gen_planet, planet, chart[planet], PLANET_DURATION, session_dir)
        for planet in m.PLANET_ORDER if planet in chart
    ]
    await asyncio.gather(*futures)
    print(f"✅ All planet tracks ready in {session_dir.name}")


# ══════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "service": "Music of the Spheres API v1.3"}


@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Step 1 — builds chart, generates main WAV + PNG (awaited).
    Step 2 — starts planet WAV generation IN BACKGROUND (not awaited).
    Returns immediately after step 1 with planet_urls included.
    Frontend fetches planet files individually — each GET is instant once ready.
    """
    # Build natal chart
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

    # Await main audio + image (parallel, ~20s total)
    await asyncio.gather(
        loop.run_in_executor(executor, _gen_audio, chart, birthdate, req.duration, session_dir),
        loop.run_in_executor(executor, _gen_image, chart, birthdate, session_dir),
    )

    # Start planet generation in background — do NOT await
    asyncio.create_task(_gen_planets_background(chart, session_dir))

    return {
        "session_id": session_id,
        "birthdate":  birthdate.strftime("%d.%m.%Y"),
        "city":       req.city,
        "tuning":     "432 Hz",
        "theory":     "Pythagorean • Kusto 1978",
        "audio_url":  f"/audio/{session_id}",
        "image_url":  f"/image/{session_id}",
        "planet_urls": {
            planet: f"/planet-audio/{session_id}/{planet.lower()}"
            for planet in m.PLANET_ORDER if planet in chart
        },
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


@app.get("/audio/{session_id}")
def get_audio(session_id: str):
    """Download full planetary WAV mix."""
    files = list((OUTPUT_DIR / session_id).glob("spheres_*.wav"))
    if not files: raise HTTPException(404, "Audio not found")
    return FileResponse(files[0], media_type="audio/wav", filename=files[0].name)


@app.get("/image/{session_id}")
def get_image(session_id: str):
    """Download PNG natal map."""
    files = list((OUTPUT_DIR / session_id).glob("spheres_*.png"))
    if not files: raise HTTPException(404, "Image not found")
    return FileResponse(files[0], media_type="image/png", filename=files[0].name)


@app.get("/planet-audio/{session_id}/{planet_name}")
def get_planet_audio(session_id: str, planet_name: str):
    """
    Download WAV for a single planet.
    Returns 202 if file not ready yet — frontend should retry after 2s.
    Returns 200 + WAV when ready.
    """
    fpath = OUTPUT_DIR / session_id / f"planet_{planet_name}.wav"
    if not fpath.exists():
        # File still being generated in background
        return JSONResponse(
            status_code=202,
            content={"status": "generating", "retry_after": 2}
        )
    return FileResponse(fpath, media_type="audio/wav")
