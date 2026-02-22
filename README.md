# Music of Spheres — API

## Запуск

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Документация: http://localhost:8000/docs

## Эндпоинты

### POST /generate
Основной эндпоинт — полный микс всех планет.

```json
{
  "year": 1990, "month": 3, "day": 15,
  "hour": 14, "minute": 30,
  "city": "Київ",
  "duration": 60
}
```

Ответ:
```json
{
  "session_id": "a1b2c3d4",
  "birthdate": "15.03.1990",
  "city": "Київ",
  "tuning": "432 Hz",
  "theory": "Pythagorean • Kusto 1978",
  "audio_url": "/audio/a1b2c3d4",
  "image_url": "/image/a1b2c3d4",
  "planets": {
    "Sun":  { "freq": 245.70, "bpm": 3.9425, "sign": "Pis", ... },
    "Moon": { "freq": 315.45, "bpm": 52.71,  "sign": "Sco", ... },
    ...
  }
}
```

### POST /generate/planets
Генерирует отдельный WAV для каждой планеты.
Используется для интерактивного интерфейса вкл/выкл.

Ответ:
```json
{
  "session_id": "a1b2c3d4",
  "planet_urls": {
    "Sun":  "/planet-audio/a1b2c3d4/sun",
    "Moon": "/planet-audio/a1b2c3d4/moon",
    ...
  },
  "chart": { ... }
}
```

### GET /audio/{session_id}
Скачать полный WAV микс.

### GET /image/{session_id}
Скачать PNG карту.

### GET /planet-audio/{session_id}/{planet}
Скачать WAV одной планеты (sun, moon, mercury, ...).

### GET /health
Статус сервера.

## Деплой на Railway

1. Создай проект на railway.app
2. Подключи репозиторий
3. Переменные окружения не нужны
4. Railway автоматически определит Python и запустит uvicorn

## Архитектура

```
POST /generate
  → build_chart() — Swiss Ephemeris, натальная карта
  → generate_audio() — Пифагорейские частоты + 432 Гц строй
  → generate_image() — Орбиты + Gaussian glow
  → session_id → файлы в /tmp/spheres/{session_id}/

POST /generate/planets
  → build_chart() — та же карта
  → make_planet_wav() × 10 — сольный WAV каждой планеты
  → 10 ссылок для фронтенда
```
