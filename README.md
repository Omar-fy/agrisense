# AgriSense

**Intelligent insight pipeline for TeleAgriCulture sensor boards**

Built for the TeleAgriCulture Live Brief ARTD2143 Module.

AgriSense takes raw sensor data from TeleAgriCulture boards, classifies it using interpretable machine learning, projects growing conditions using OpenWeatherMap forecast data; detecting anomalies in real time and surfacing everything through a plain HTML dashboard and REST API.

---

## What it does

- **Pulls** sensor readings from the TeleAgriCulture API for any Kit ID
- **Classifies** each time window into an agronomic state (optimal, heat stress, drought risk, etc.) using a shallow Decision Tree (max depth 4)
- **Detects anomalies** using a rolling z-score against the board's learned baseline
- **Forecasts** 5-day indoor conditions using the board's learned indoor/outdoor temperature delta and OpenWeatherMap data
- **Benchmarks** boards against each other using peer statistical profiles
- **Stores** all readings in a local SQLite database for persistent multi-board tracking
- **Discovers** public TeleAgriCulture boards automatically by probing the API
- **Uploads** cached readings to the TeleAgriCulture platform via a WiFi handshake workflow

---

## Project structure

```
agrisense/
├── pipeline/
│   ├── pipeline.py       Core ML pipeline (18 functions)
│   └── __init__.py
├── main.py               FastAPI backend; has all API endpoints
├── database.py           SQLite storage layer
├── discovery.py          Automatic board discovery
├── upload.py             WiFi handshake and upload-to-TeleAgriCulture layer
├── dashboard.html        Plain HTML/JS frontend (Rudimentary front to prototype with, the project is part of a team effort and will be merged with a more presentable UI)
├── requirements.txt
└── .env.example
```

---

## Setup

### Requirements

- Python 3.10+
- A TeleAgriCulture board with a Kit ID from [kits.teleagriculture.org](https://kits.teleagriculture.org)
- An OpenWeatherMap API key (free tier) for forecast features

### Install

```bash
git clone https://github.com/YOUR_USERNAME/agrisense.git

# Or just download the zip from the repo

cd agrisense

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Open .env and add your OWM_API_KEY
```

### Run

```bash
uvicorn main:app --reload --port 8000
```

API docs available at: **http://localhost:8000/docs**

### Open the dashboard (prototype)

```bash
# Option 1 — Python built-in server (recommended and always worked)
python3 -m http.server 8080
# then open http://localhost:8080/dashboard.html

# Option 2 — VS Code Live Server (some hiccups that may be hardware limitations .. will rectify)
# Right-click dashboard.html → Open with Live Server

# Option 3 — open directly in browser (may have CORS issues but still working on them)
open dashboard.html
```

---

## Usage

### Using the demo board (Kit 1001)

Kit 1001 is the Schmiede festival demo board in Salzburg, Austria. It has 1,219 real temperature readings from August–September 2019 and is always available without any authentication.

```bash
# Sync readings into local database
curl -X POST "http://localhost:8000/boards/sync/1001?sensor_name=ftTemp"

# Get insights
curl "http://localhost:8000/kit/1001/insights?sensor_name=ftTemp&lat=47.7981&lon=13.0456"

# Get 5-day forecast (requires OWM_API_KEY in .env)
curl "http://localhost:8000/kit/1001/forecast?sensor_name=ftTemp&lat=47.7981&lon=13.0456"
```

### Discovering public boards

```bash
# Fast — probe a list of known Kit IDs
curl -X POST http://localhost:8000/boards/discover \
  -H "Content-Type: application/json" \
  -d '{"mode": "seeds"}'

# Slower — scan a range
curl -X POST http://localhost:8000/boards/discover \
  -H "Content-Type: application/json" \
  -d '{"mode": "range", "start_id": 1000, "end_id": 1050}'

# Check what was found
curl http://localhost:8000/boards
```

### WiFi handshake — offline to online upload

The TeleAgriCulture board creates its own WiFi access point during setup. AgriSense uses that same WiFi channel to pull readings from the board directly and cache them locally, then uploads to the TeleAgriCulture platform when internet is available.

```bash
# 1. Connect your laptop/phone to the board's WiFi
#    SSID: TeleAgriCulture Board
#    Password: enter123

# 2. Check the board is reachable
curl http://localhost:8000/upload/board/status

# 3. Run the handshake — pull reading and cache locally
curl -X POST http://localhost:8000/upload/handshake \
  -H "Content-Type: application/json" \
  -d '{"kit_id": "YOUR_KIT_ID"}'

# 4. When back on normal internet, upload cached readings
curl -X POST http://localhost:8000/upload/flush \
  -H "Content-Type: application/json" \
  -d '{"kit_id": "YOUR_KIT_ID", "sensor_name": "ftTemp", "api_token": "YOUR_TOKEN"}'

# 5. Check upload queue status
curl "http://localhost:8000/upload/queue/YOUR_KIT_ID?sensor_name=ftTemp"
```

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/` | Health check |
| GET  | `/boards` | List all registered boards |
| POST | `/boards/discover` | Discover public boards (seeds or range) |
| POST | `/boards/sync/{kit_id}` | Fetch and store readings for one board |
| POST | `/boards/sync/all` | Sync all registered boards |
| GET  | `/boards/{kit_id}/readings` | Query stored readings |
| GET  | `/kit/{kit_id}/insights` | Full pipeline — insights, anomalies, peers |
| GET  | `/kit/{kit_id}/forecast` | 5-day indoor forecast |
| GET  | `/kit/{kit_id}/alerts` | Alert queue |
| POST | `/kit/{kit_id}/alerts/check` | Check for new alerts |
| DELETE | `/kit/{kit_id}/alerts` | Clear alerts |
| GET  | `/upload/board/status` | Check if board AP is reachable |
| POST | `/upload/handshake` | WiFi handshake — pull and cache board readings |
| POST | `/upload/flush` | Upload cached readings to TeleAgriCulture |
| POST | `/upload/queue/{kit_id}` | Queue stored readings for upload |
| GET  | `/upload/queue/{kit_id}` | Upload queue status |
| GET  | `/db/summary` | Database stats |
| GET  | `/peers` | Peer profiles for benchmarking |

---

## Pipeline stages

```
TeleAgriCulture API / Board WiFi AP
          │
  fetch_sensor_data()     Paginated API fetch with cursor handling
          │
  clean()                 Drop nulls, physical bounds, dedup timestamps
  calibrate_soil_moisture() Convert raw ADC → 0-100% if needed
          │
  featurise()             Lagged features + rolling stats (from Schmiede notebook)
          │
  build_classifier()      Decision Tree max_depth=4 — interpretable rules
  classify_windows()      Label each time window with agronomic state
          │
          ├── detect_anomalies()     Rolling z-score, threshold 2.5
          ├── compute_deviation()    Board mean vs OWM outdoor baseline
          ├── compare_to_peers()     Cross-board percentile benchmarking
          │
  generate_summary()      Plain-language insight for non-technical users
          │
  FastAPI → dashboard.html
```

### Agronomic states

| State | Sensor | Condition |
|-------|--------|-----------|
| `optimal` | Temperature | 15–28°C |
| `heat_stress` | Temperature | >28°C |
| `cold_stress` | Temperature | <5°C |
| `cool` | Temperature | 5–15°C |
| `optimal` | Soil moisture | 40–70% |
| `drought_risk` | Soil moisture | <20% |
| `waterlogged` | Soil moisture | >80% |

---

## Data flow — offline to online

```
[Board collects readings on-device / SD card]
          │
          (connect to board WiFi AP)
[AgriSense handshake pull → local SQLite cache]
          │
          (back on internet)
[AgriSense upload flush → kits.teleagriculture.org SQL/PHP database]
          │
[TeleAgriCulture platform — available for wider scientific use]
```

---

## Frontend

`dashboard.html` is a single self-contained file. Open it in any browser or serve it with a static file server. It connects to the FastAPI backend at `http://localhost:8000`.

The frontend is plain HTML/CSS/JS and could be trasnformed into standard JSON to work with any HTTP client and/or a React frontend which I will be discussing with my team on.

---

## Environment variables

Copy `.env.example` to `.env`:

```
OWM_API_KEY=your_openweathermap_key_here
```

Without an OWM key the forecast tab returns an error but all other features work.

---

## Partner context

- **Hardware:** TeleAgriCulture Board V2.1
- **Demo board:** Kit 1001, Schmiede festival, Salzburg 2019
- **Original notebook:** TeleAgriCulture_Schmiede.ipynb (pipeline.py is a full refactor of this)

---

## References

- Miller, T. et al. (2025) The IoT and AI in agriculture: a systematic review. *Sensors*, 25(12).
- Fuentes-Penailillo, F. et al. (2024) Transformative technologies in digital agriculture. *Journal of Sensor and Actuator Networks*, 13(4).
- TeleAgriCulture (2023) Platform documentation. [teleagriculture.org](https://teleagriculture.org)
