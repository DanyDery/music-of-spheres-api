"""
Music of the Spheres — Core Generator
══════════════════════════════════════════════════════════
Sources:
  - Pythagoras / Iamblichus: three sacred intervals —
    perfect fourth (4:3), perfect fifth (3:2), octave (2:1).
    Three spheres: Moon, Sun, Stars.
  - "Most perfect harmony" (Nicomachus): proportion 6:8:9:12
  - Hans Cousto (1978): base frequencies from orbital periods
  - Tuning: 432 Hz — natural tuning system used by Cousto
  - Timbre: string (Iamblichus: Pythagoras taught to reproduce
    cosmic sounds "using string instruments" — monochord)
  - Rhythm: planetary periods → BPM via octave method (Cousto)
  - Earth drone: foundation of the system
    (Robert Fludd's Celestial Monochord — Terra at the base)

Install:
  pip install kerykeion numpy pillow geopy timezonefinder certifi
"""

import ssl, certifi, os
os.environ['SSL_CERT_FILE']      = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from kerykeion import AstrologicalSubject
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
import wave, struct, math, json
from datetime import date
from pathlib import Path


# ══════════════════════════════════════════════
# 1. CONSTANTS
# ══════════════════════════════════════════════

# Planetary base frequencies (Hans Cousto, 1978)
PLANET_BASE_FREQ = {
    "Sun":     126.22, "Moon":    210.42, "Mercury": 141.27,
    "Venus":   221.23, "Earth":   194.18, "Mars":    144.72,
    "Jupiter": 183.58, "Saturn":  147.85, "Uranus":  207.36,
    "Neptune": 211.44, "Pluto":   140.25,
}

# Orbital periods in days → planetary rhythm
PLANET_PERIODS = {
    "Sun":     365.25,  "Moon":     27.32,  "Mercury":  87.97,
    "Venus":   224.70,  "Mars":    686.97,  "Jupiter": 4332.59,
    "Saturn": 10759.22, "Uranus": 30688.50, "Neptune": 60182.00,
    "Pluto":  90560.00,
}

# 12 zodiac signs = 12 chromatic semitones (Pythagoras)
SIGN_SEMITONE = {
    "Ari":0, "Tau":1, "Gem":2, "Can":3,
    "Leo":4, "Vir":5, "Lib":6, "Sco":7,
    "Sag":8, "Cap":9, "Aqu":10,"Pis":11,
}

PLANET_COLOR = {
    "Sun":    (255,200, 50), "Moon":   (200,210,255),
    "Mercury":(180,180,200), "Venus":  (255,180,120),
    "Mars":   (220, 80, 60), "Jupiter":(200,160,100),
    "Saturn": (210,190,130), "Uranus": (100,220,210),
    "Neptune":( 60,100,220), "Pluto":  (140, 60,180),
}

# Appearance order — personal planets first, outer planets last
# Sun and Moon are primary (Pythagoras: three spheres = Moon, Sun, Stars)
PLANET_ORDER = ["Sun","Moon","Mercury","Venus","Mars",
                "Jupiter","Saturn","Uranus","Neptune","Pluto"]

# Planet weight in the mix — Sun and Moon are primary per Pythagoras
PLANET_WEIGHT = {
    "Sun":    1.0,   # primary sphere
    "Moon":   0.9,   # primary sphere
    "Mercury":0.65,
    "Venus":  0.65,
    "Mars":   0.60,
    "Jupiter":0.55,
    "Saturn": 0.55,
    "Uranus": 0.45,
    "Neptune":0.45,
    "Pluto":  0.40,
}

# 432 Hz tuning — natural tuning system (Cousto)
# Standard A=440 Hz was adopted in 1939; Cousto worked in A=432 Hz
TUNING_432 = 432 / 440

SAMPLE_RATE = 44100
CACHE_FILE  = Path("cities_cache.json")


# ══════════════════════════════════════════════
# 2. PYTHAGOREAN FORMULAS
# ══════════════════════════════════════════════

def calc_freq(planet: str, sign: str, degree: float) -> float:
    """
    Pythagorean chromatic frequency:
      base  = planet base frequency (Cousto)
      sign  → semitone 0..11 (position in zodiacal octave)
      degree → microtonal tuning within semitone (0..29°)

    freq = base × 2^(semitone/12) × 2^(degree/29/12) × 432/440
    """
    base     = PLANET_BASE_FREQ[planet]
    semitone = SIGN_SEMITONE.get(sign[:3], 0)
    freq     = base * (2**(semitone/12)) * (2**(degree/29/12))
    return round(freq * TUNING_432, 2)

def calc_bpm(planet: str) -> float:
    """
    Planetary rhythm: orbital period → BPM via octave method.
    Same principle Cousto used for frequencies.
    """
    bpm = (1.0 / PLANET_PERIODS[planet]) * 1440
    while bpm < 0.5: bpm *= 2
    while bpm > 120: bpm /= 2
    return bpm


# ══════════════════════════════════════════════
# 3. GEOCODING WITH CACHE
# ══════════════════════════════════════════════

def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}

def save_cache(cache: dict):
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2))

def resolve_city(city: str) -> tuple:
    """City in any language → (lat, lng, timezone). Cached."""
    cache = load_cache()
    key   = city.strip().lower()
    if key in cache:
        d = cache[key]
        print(f"  📍 {d['address']} (from cache)")
        return d['lat'], d['lng'], d['tz']

    geolocator = Nominatim(user_agent="music_of_spheres")
    tf = TimezoneFinder()
    location = geolocator.geocode(city)
    if not location:
        for suffix in [", Ukraine",", Russia",", Poland",", Germany"]:
            location = geolocator.geocode(city + suffix)
            if location: break

    if not location:
        raise ValueError(
            f"City '{city}' not found.\n"
            f"  Try: 'Kharkiv, Ukraine' or 'Warsaw, Poland'"
        )

    lat = location.latitude
    lng = location.longitude
    tz  = tf.timezone_at(lat=lat, lng=lng)
    if not tz:
        raise ValueError(f"Could not determine timezone for '{city}'.")

    cache[key] = {"address":location.address,"lat":lat,"lng":lng,"tz":tz}
    save_cache(cache)
    print(f"  📍 {location.address}")
    print(f"     {lat:.4f}, {lng:.4f}  |  {tz}")
    return lat, lng, tz


# ══════════════════════════════════════════════
# 4. NATAL CHART
# ══════════════════════════════════════════════

def planet_strength(deg: float) -> float:
    """Planet strength: max at cusps (0° and 29°), min at midpoint (15°)."""
    return 1.0 - 0.5 * abs(deg - 15) / 15

def build_chart(year, month, day, hour, minute, city) -> dict:
    lat, lng, tz = resolve_city(city)
    subj = AstrologicalSubject(
        "chart", year, month, day, hour, minute,
        lng=lng, lat=lat, tz_str=tz, city=city
    )
    raw = {
        "Sun":subj.sun,"Moon":subj.moon,"Mercury":subj.mercury,
        "Venus":subj.venus,"Mars":subj.mars,"Jupiter":subj.jupiter,
        "Saturn":subj.saturn,"Uranus":subj.uranus,
        "Neptune":subj.neptune,"Pluto":subj.pluto,
    }
    chart = {}
    for name, obj in raw.items():
        deg     = float(obj.position)
        abs_deg = float(obj.abs_pos)
        sign    = obj.sign[:3]
        chart[name] = {
            "freq":      calc_freq(name, sign, deg),
            "base_freq": PLANET_BASE_FREQ[name],
            "bpm":       calc_bpm(name),
            "beat_hz":   calc_bpm(name) / 60.0,
            "strength":  planet_strength(deg),
            "weight":    PLANET_WEIGHT[name],
            "sign":      obj.sign,
            "sign_short":sign,
            "deg":       deg,
            "abs_deg":   abs_deg,
            "color":     PLANET_COLOR[name],
        }

    print(f"\n  {'Planet':<10} {'Sign':<13} {'Degree':>7}°"
          f"  {'Frequency':>9} Hz  {'Rhythm':>7} BPM  {'Strength':>8}")
    print(f"  {'─'*68}")
    for p, d in chart.items():
        print(f"  {p:<10} {d['sign']:<13} {d['deg']:>7.2f}°"
              f"  {d['freq']:>9.2f} Hz  {d['bpm']:>7.4f} BPM"
              f"  {d['strength']:>8.2f}")
    return chart


# ══════════════════════════════════════════════
# 5. AUDIO
# ══════════════════════════════════════════════

def string_tone(freq: float, t: np.ndarray, phase: float) -> np.ndarray:
    """
    String timbre — Fourier series of sawtooth wave (6 harmonics).
    Source: Iamblichus — Pythagoras taught to reproduce cosmic sounds
    'using string instruments' (monochord).
    """
    result = np.zeros_like(t)
    for n in range(1, 7):
        result += ((-1)**(n+1)) * np.sin(2*math.pi*freq*n*t + phase) / n
    return result * (2/math.pi)

def reverb(signal, decay=0.35, delay_ms=90, num_echoes=5):
    """Simple reverb — series of decaying echoes. Creates cosmic space feel."""
    delay_samples = int(SAMPLE_RATE * delay_ms / 1000)
    result = signal.copy()
    for i in range(1, num_echoes+1):
        shift   = delay_samples * i
        amp     = decay ** i
        shifted = np.zeros_like(signal)
        if shift < len(signal):
            shifted[shift:] = signal[:-shift]
        result += amp * shifted
    peak = np.max(np.abs(result))
    if peak > 0:
        result = result / peak * np.max(np.abs(signal))
    return result

def envelope(sig, attack=3.0, release=4.0):
    n = len(sig)
    env = np.ones(n)
    atk = int(SAMPLE_RATE * attack)
    rel = int(SAMPLE_RATE * release)
    env[:atk]  = np.linspace(0, 1, atk)
    env[-rel:] = np.linspace(1, 0, rel)
    return sig * env

def generate_audio(chart: dict, birthdate: date, duration: int = 60) -> str:
    """
    Final Music of the Spheres according to sources:

    Three Pythagorean intervals (Pythagoras / Iamblichus):
      - Perfect fifth  (3:2) = freq × 1.500
      - Perfect fourth (4:3) = freq × 1.333
      - Octave         (2:1) = freq × 2.000

    Sphere hierarchy: Sun and Moon are primary, others secondary.
    Proportion 6:8:9:12 — basis of "most perfect harmony".

    Gradual entry: personal planets first,
    outer planets enter later — like the birth of a chart.

    Reverb: cosmic space.
    Dynamics: each planet breathes at its own rhythm.
    """
    n_samples = int(SAMPLE_RATE * duration)
    t     = np.linspace(0, duration, n_samples, endpoint=False)
    left  = np.zeros(n_samples)
    right = np.zeros(n_samples)

    n_planets = len(PLANET_ORDER)

    for idx, planet in enumerate(PLANET_ORDER):
        if planet not in chart: continue
        d        = chart[planet]
        freq     = d["freq"]
        strength = d["strength"]
        weight   = d["weight"]
        abs_deg  = d["abs_deg"]
        beat_hz  = d["beat_hz"]
        phase    = (abs_deg/360.0) * 2*math.pi

        # Gradual entry — Sun and Moon from the start,
        # outer planets enter gradually up to 40% of track
        entry_time = (idx / n_planets) * 0.4 * duration
        fade_dur   = duration * 0.12

        entry_env = np.zeros(n_samples)
        e_start = int(entry_time * SAMPLE_RATE)
        e_end   = min(int((entry_time+fade_dur)*SAMPLE_RATE), n_samples)
        if e_start < n_samples:
            fade_len = e_end - e_start
            entry_env[e_start:e_end] = np.linspace(0, 1, fade_len)
            entry_env[e_end:] = 1.0

        # Dynamics: breathing + planetary pulse
        breath_period = max(10.0, 60.0/beat_hz*0.3)
        breath = 0.75 + 0.25 * np.sin(2*math.pi/breath_period*t + phase*0.08)
        lfo    = 0.65 + 0.35 * np.sin(2*math.pi*beat_hz*t)
        amp_env = entry_env * breath * lfo

        # Amplitude = base × planet weight × chart strength
        amp = 0.13 * weight * strength

        # Three Pythagorean intervals
        tone  = amp * amp_env * string_tone(freq,        t, phase)
        quint = amp * 0.35 * amp_env * string_tone(freq*1.500, t, phase)  # fifth
        quart = amp * 0.25 * amp_env * string_tone(freq*1.333, t, phase)  # fourth
        octv  = amp * 0.15 * amp_env * string_tone(freq*2.000, t, phase)  # octave

        # Stereo panning from ecliptic position
        pan   = 0.5 + 0.45*math.sin(math.radians(abs_deg))
        signal = tone + quint + quart + octv
        left  += signal * (1-pan)
        right += signal * pan

    # Earth drone — foundation of the system
    # Source: Fludd's Celestial Monochord — Terra at the base
    earth_freq = PLANET_BASE_FREQ["Earth"] * TUNING_432
    earth_bpm  = calc_bpm("Sun")
    earth_lfo  = 0.7 + 0.3*np.sin(2*math.pi*(earth_bpm/60)*t)
    earth = 0.06 * earth_lfo * string_tone(earth_freq, t, 0)
    left  += earth; right += earth

    # Binaural beat: Sun ↔ Moon
    # Primary spheres of Pythagoras create a beat between them
    sun_f  = chart["Sun"]["freq"]
    moon_f = chart["Moon"]["freq"]
    beat   = abs(sun_f - moon_f)
    sp     = (chart["Sun"]["abs_deg"]  /360.0)*2*math.pi
    mp     = (chart["Moon"]["abs_deg"] /360.0)*2*math.pi
    left  += 0.07 * np.sin(2*math.pi*sun_f*t + sp)
    right += 0.07 * np.sin(2*math.pi*(sun_f+beat)*t + mp)

    # Reverb (different L/R delay → stereo width)
    left  = reverb(left,  decay=0.35, delay_ms=90,  num_echoes=5)
    right = reverb(right, decay=0.35, delay_ms=115, num_echoes=5)

    # Master envelope
    master = np.ones(n_samples)
    atk = int(SAMPLE_RATE*3.0); rel = int(SAMPLE_RATE*5.0)
    master[:atk]  = np.linspace(0, 1, atk)
    master[-rel:] = np.linspace(1, 0, rel)
    left  *= master; right *= master

    peak = max(np.max(np.abs(left)), np.max(np.abs(right)))
    if peak > 0:
        left, right = left/peak*0.88, right/peak*0.88

    fname = f"spheres_{birthdate.strftime('%Y%m%d')}.wav"
    with wave.open(fname, 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        for l, r in zip(left, right):
            wf.writeframes(struct.pack('<hh',
                int(np.clip(l,-1,1)*32767),
                int(np.clip(r,-1,1)*32767)))

    print(f"\n  🎵 {fname}")
    return fname


# ══════════════════════════════════════════════
# 6. VISUAL
# ══════════════════════════════════════════════

def add_glow(canvas, px, py, color, strength, size):
    glow_r = int(10+12*strength)
    x0=max(0,int(px)-glow_r); x1=min(size,int(px)+glow_r+1)
    y0=max(0,int(py)-glow_r); y1=min(size,int(py)+glow_r+1)
    xs=np.arange(x0,x1); ys=np.arange(y0,y1)
    xx,yy=np.meshgrid(xs,ys)
    dist=np.sqrt((xx-px)**2+(yy-py)**2)
    glow=np.exp(-0.5*(dist/(glow_r*0.38))**2)*strength*0.85
    for ch,c in enumerate(color):
        canvas[y0:y1,x0:x1,ch] += glow*(c/255.0)
    core_r=max(2,int(2+3*strength))
    xc0=max(0,int(px)-core_r); xc1=min(size,int(px)+core_r+1)
    yc0=max(0,int(py)-core_r); yc1=min(size,int(py)+core_r+1)
    xs2=np.arange(xc0,xc1); ys2=np.arange(yc0,yc1)
    xx2,yy2=np.meshgrid(xs2,ys2)
    dist2=np.sqrt((xx2-px)**2+(yy2-py)**2)
    core=np.clip(1-dist2/max(core_r,1),0,1)**1.5
    for ch,c in enumerate(color):
        canvas[yc0:yc1,xc0:xc1,ch] += core*(c/255.0)*1.3

def generate_image(chart: dict, birthdate: date, size: int = 1024) -> str:
    seed=int(birthdate.strftime('%Y%m%d'))
    rng=np.random.default_rng(seed)
    cx,cy=size//2,size//2
    canvas=np.zeros((size,size,3),dtype=np.float32)

    # Stars — three layers
    for _ in range(3500):
        x=int(rng.integers(0,size)); y=int(rng.integers(0,size))
        if 0<=y<size and 0<=x<size:
            canvas[y,x] += rng.uniform(0.18,0.75)
    for _ in range(400):
        x=int(rng.integers(0,size)); y=int(rng.integers(0,size))
        br=rng.uniform(0.55,1.0)
        if 0<=y<size and 0<=x<size:
            canvas[y,x]=np.minimum(1.0,canvas[y,x]+br)
            for nx2,ny2 in [(-1,0),(1,0),(0,-1),(0,1)]:
                if 0<=y+ny2<size and 0<=x+nx2<size:
                    canvas[y+ny2,x+nx2]+=br*0.35
    for _ in range(60):
        x=int(rng.integers(20,size-20)); y=int(rng.integers(20,size-20))
        br=rng.uniform(0.85,1.0)
        canvas[y,x]=1.0
        for length in range(1,6):
            fade=br*(1-length/6)**1.5
            for dx2,dy2 in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx2=x+dx2*length; ny2=y+dy2*length
                if 0<=ny2<size and 0<=nx2<size:
                    canvas[ny2,nx2]+=fade
        for length in range(1,4):
            fade=br*(1-length/4)**2
            for dx2,dy2 in [(-1,-1),(1,-1),(-1,1),(1,1)]:
                nx2=x+dx2*length; ny2=y+dy2*length
                if 0<=ny2<size and 0<=nx2<size:
                    canvas[ny2,nx2]+=fade*0.5

    # Planet glows
    base_r=44; step_r=44
    planet_coords={}
    for i,planet in enumerate(PLANET_ORDER):
        if planet not in chart: continue
        d=chart[planet]; r=base_r+i*step_r
        angle=math.radians(d["abs_deg"]-90)
        px=cx+r*math.cos(angle); py=cy+r*math.sin(angle)
        planet_coords[planet]=(px,py)
        add_glow(canvas,px,py,d["color"],d["strength"],size)

    # Sun at center (primary sphere)
    for gr in range(38,0,-1):
        y0=max(0,cy-gr); y1=min(size,cy+gr+1)
        x0=max(0,cx-gr); x1=min(size,cx+gr+1)
        ys=np.arange(y0,y1); xs=np.arange(x0,x1)
        xx,yy=np.meshgrid(xs,ys)
        dist=np.sqrt((xx-cx)**2+(yy-cy)**2)
        g=np.exp(-0.5*(dist/max(gr*0.4,1))**2)*0.06
        for ch,c in enumerate(PLANET_COLOR["Sun"]):
            canvas[y0:y1,x0:x1,ch]+=g*(c/255.0)

    img_base=Image.fromarray(
        np.clip(canvas,0,1).__mul__(255).astype(np.uint8),'RGB')

    # Orbital rings
    overlay=Image.new('RGBA',(size,size),(0,0,0,0))
    draw_o=ImageDraw.Draw(overlay,'RGBA')
    for i,planet in enumerate(PLANET_ORDER):
        if planet not in chart: continue
        d=chart[planet]; r=base_r+i*step_r
        weight=d["weight"]
        ring_a=int(120+135*weight*d["strength"])
        ring_w=2 if weight>=0.9 else 1  # Sun and Moon rings are thicker
        draw_o.ellipse([cx-r,cy-r,cx+r,cy+r],
                       outline=d["color"]+(ring_a,),width=ring_w)

    img_base=Image.alpha_composite(
        img_base.convert('RGBA'),overlay).convert('RGB')
    img_base=img_base.filter(ImageFilter.GaussianBlur(radius=0.5))
    draw_f=ImageDraw.Draw(img_base)

    try:
        font_sm=ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",11)
        font_dt=ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",13)
    except:
        font_sm=ImageFont.load_default(); font_dt=font_sm

    # Planet labels with Pythagorean frequency
    for planet,(px,py) in planet_coords.items():
        d=chart[planet]
        label=f"{planet}  {d['freq']} Hz"
        dx=px-cx; dy=py-cy
        dist=math.hypot(dx,dy)
        nx,ny=(dx/dist,dy/dist) if dist>0 else (1,0)
        tx=px+nx*16; ty=py+ny*16
        draw_f.text((tx+1,ty+1),label,fill=(0,0,0),font=font_sm,anchor="lm")
        draw_f.text((tx,ty),label,fill=d["color"],font=font_sm,anchor="lm")

    # Bottom label: Sun sign, Moon sign, date
    label_bot=(f"☀ {chart['Sun']['sign']}  ☽ {chart['Moon']['sign']}"
               f"  ·  {birthdate.strftime('%d.%m.%Y')}")
    draw_f.text((cx,size-22),label_bot,
                fill=(190,190,210),font=font_dt,anchor="mm")

    fname=f"spheres_{birthdate.strftime('%Y%m%d')}.png"
    img_base.save(fname)
    print(f"  🖼️  {fname}")
    return fname


# ══════════════════════════════════════════════
# 7. RUN — EDIT THESE VALUES
# ══════════════════════════════════════════════

if __name__ == "__main__":

    # ── Your birth data ───────────────────────
    BIRTH_YEAR = 1990
    BIRTH_MON  = 3
    BIRTH_DAY  = 15
    BIRTH_HOUR = 14    # if unknown — use 12
    BIRTH_MIN  = 30
    CITY       = "Kyiv"   # any language: "Kharkiv", "Warsaw", "Berlin"
    DURATION   = 60       # seconds; for production use 180-300
    # ──────────────────────────────────────────

    print(f"\n🌌 Music of the Spheres")
    print(f"   Date:  {BIRTH_DAY:02d}.{BIRTH_MON:02d}.{BIRTH_YEAR}"
          f"  {BIRTH_HOUR:02d}:{BIRTH_MIN:02d}")
    print(f"   City:  {CITY}")
    print("═"*55)

    birthdate  = date(BIRTH_YEAR, BIRTH_MON, BIRTH_DAY)
    chart      = build_chart(BIRTH_YEAR, BIRTH_MON, BIRTH_DAY,
                             BIRTH_HOUR, BIRTH_MIN, CITY)
    audio_file = generate_audio(chart, birthdate, DURATION)
    image_file = generate_image(chart, birthdate)

    print("\n"+"═"*55)
    print("✅ Done!")
    print(f"   🎵 {audio_file}")
    print(f"   🖼️  {image_file}")
