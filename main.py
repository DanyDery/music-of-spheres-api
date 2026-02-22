"""
Музыка Сфер — FastAPI бэкенд
════════════════════════════
Эндпоинты:
  POST /generate            → карта + WAV + PNG
  GET  /audio/{id}          → полный WAV
  GET  /image/{id}          → PNG
  POST /generate/planets    → 10 отдельных WAV
  GET  /planet-audio/{id}/{planet} → WAV одной планеты
  GET  /health              → статус

Установка:
  pip install fastapi uvicorn kerykeion numpy pillow
              geopy timezonefinder certifi

Запуск:
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
    title="Music of Spheres API",
    description="Pythagorean planetary matrix generator • Kusto 1978 • 432 Hz",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # в продакшене → конкретный домен фронтенда
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("/tmp/spheres")
OUTPUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════
# СХЕМЫ
# ══════════════════════════════════════════════

class GenerateRequest(BaseModel):
    year:     int = Field(..., ge=1900, le=2100, example=1990)
    month:    int = Field(..., ge=1,    le=12,   example=3)
    day:      int = Field(..., ge=1,    le=31,   example=15)
    hour:     int = Field(12,  ge=0,    le=23,   example=14)
    minute:   int = Field(0,   ge=0,    le=59,   example=30)
    city:     str = Field(..., min_length=2,     example="Київ")
    duration: int = Field(60,  ge=10,   le=600,  example=60)


# ══════════════════════════════════════════════
# ХЕЛПЕРЫ
# ══════════════════════════════════════════════

def build_session(req: GenerateRequest):
    """Строит карту, создаёт сессию."""
    try:
        birthdate = date(req.year, req.month, req.day)
    except ValueError as e:
        raise HTTPException(400, f"Неверная дата: {e}")
    try:
        chart = m.build_chart(
            req.year, req.month, req.day,
            req.hour, req.minute, req.city
        )
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Ошибка геокодинга: {e}")

    session_id  = str(uuid.uuid4())[:8]
    session_dir = OUTPUT_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    return session_id, chart, birthdate, session_dir


def make_planet_wav(planet: str, d: dict,
                    duration: int, session_dir: Path) -> Path:
    """Генерирует сольный WAV одной планеты с тремя интервалами."""
    freq    = d["freq"]
    abs_deg = d["abs_deg"]
    beat_hz = d["beat_hz"]
    phase   = (abs_deg / 360.0) * 2 * math.pi
    amp     = 0.55

    t    = np.linspace(0, duration,
                       int(m.SAMPLE_RATE * duration), endpoint=False)
    lfo  = 0.65 + 0.35 * np.sin(2 * math.pi * beat_hz * t)
    tone = amp * lfo * m.string_tone(freq,        t, phase)
    q5   = amp * 0.35 * lfo * m.string_tone(freq * 1.500, t, phase)  # квинта
    q4   = amp * 0.25 * lfo * m.string_tone(freq * 1.333, t, phase)  # кварта
    oct_ = amp * 0.15 * lfo * m.string_tone(freq * 2.000, t, phase)  # октава
    sig  = tone + q5 + q4 + oct_

    sig  = m.envelope(sig, attack=2.0, release=3.0)
    sig  = m.reverb(sig, decay=0.3, delay_ms=80, num_echoes=4)
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig = sig / peak * 0.85

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
# ЭНДПОИНТЫ
# ══════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "service": "Music of Spheres API v1.0"}


@app.post("/generate")
def generate(req: GenerateRequest):
    """
    Основной эндпоинт.
    Строит натальную карту, генерирует полный WAV и PNG.
    Возвращает данные карты + ссылки на файлы.
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
    Генерирует отдельный WAV для каждой из 10 планет.
    Фронтенд использует эти треки для интерактивного вкл/выкл.
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
    files = list((OUTPUT_DIR / session_id).glob("spheres_*.wav"))
    if not files:
        raise HTTPException(404, "Аудио не найдено")
    return FileResponse(files[0], media_type="audio/wav",
                        filename=files[0].name)


@app.get("/image/{session_id}")
def get_image(session_id: str):
    files = list((OUTPUT_DIR / session_id).glob("spheres_*.png"))
    if not files:
        raise HTTPException(404, "Изображение не найдено")
    return FileResponse(files[0], media_type="image/png",
                        filename=files[0].name)


@app.get("/planet-audio/{session_id}/{planet_name}")
def get_planet_audio(session_id: str, planet_name: str):
    fpath = OUTPUT_DIR / session_id / f"planet_{planet_name}.wav"
    if not fpath.exists():
        raise HTTPException(404, f"Планета '{planet_name}' не найдена")
    return FileResponse(fpath, media_type="audio/wav")
