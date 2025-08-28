# Stage 13 — Productization Demo

## How to run
1) pip install -r requirements.txt
2) python app.py  # starts API at http://127.0.0.1:5000

## Endpoints
- POST /predict  (JSON: {"features": [[value]]})
- GET  /predict/<x>
- GET  /predict/<x>/<y>
- GET  /plot  (returns PNG)
