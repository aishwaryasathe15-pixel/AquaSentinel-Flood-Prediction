# AquaSentinel AI — Flood Risk Prediction Website

A beginner-friendly Flask + Python + scikit-learn project. It includes user registration/login, CSV upload, Random Forest flood-risk classification, prediction history, a dashboard with charts, and risk guidance.

## Features
- Register/login/logout; passwords are stored as hashes in SQLite.
- Predict **Low / Medium / High / Extreme** risk using rainfall, water level, humidity, temperature, elevation, and previous flood occurrence.
- Upload CSV data. A labelled CSV containing a `flood_risk` column can retrain the model.
- Dashboard graphs: historical rainfall, water level, and distribution of risk categories.
- Prediction history for each signed-in user.
- Download included sample dataset.
- Responsive pages and early-warning guidance.

## Important academic / safety note
The included dataset is **synthetic demonstration data**, not real observed flood records. Its model accuracy is only a software-demo metric and must not be treated as real-world forecast performance. This is not an official warning service and should not be used for evacuation or life-safety decisions. A real deployment needs validated local historical observations, rigorous temporal/geographic testing, calibrated probabilities, and official weather/hydrology data.

## 1. Open in VS Code
1. Extract the ZIP file.
2. Open VS Code → **File → Open Folder** → select `flood_risk_prediction_system`.
3. Open **Terminal → New Terminal**.
4. Confirm Python is installed: `python --version` (Windows may use `py --version`).

## 2. Create a virtual environment
Windows PowerShell:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```
If PowerShell blocks activation, use Command Prompt terminal:
```bat
.venv\Scripts\activate.bat
```
If `py` is not recognized, try `python -m venv .venv`.

## 3. Install libraries
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Run the website
```bash
python app.py
```
Open the URL printed in the terminal, normally `http://127.0.0.1:5000`.

## 5. First use
1. Select **Create account** and register with a username, email, and password of at least 8 characters.
2. Log in.
3. Open **Predict Risk**, enter environmental values, and submit.
4. Open **Dashboard** to view charts and recent predictions.
5. Open **Upload Data** to upload your historical CSV.
6. Use **Download sample CSV** to inspect the exact format.

## CSV format
Required numeric feature columns (headers can use common alternatives like `rainfall`, `water_level`, `humidity`, `temperature`, `elevation`):
- `rainfall_mm`
- `water_level_m`
- `humidity_pct`
- `temperature_c`
- `elevation_m`
- `previous_flood` (0 or 1)

To **retrain** the model, include a target column `flood_risk` with labels `Low`, `Medium`, `High`, or `Extreme`. A binary target `0/1` is mapped to `Low/High` for a simple demonstration. At least 20 valid rows and at least 2 classes are required. If the target column is omitted, the CSV is saved for visualisation only and the existing model remains active.

## Troubleshooting
- `ModuleNotFoundError`: activate `.venv`, then run `pip install -r requirements.txt`.
- Port 5000 already in use: close the other app or change `app.run(debug=True, port=5001)`.
- If the model file is corrupted or you changed dataset schema, stop the server and delete `flood_model.joblib`; restart.
- Charts load Chart.js from a CDN, so the charts need an internet connection. The rest of the local website runs on your computer.
- For production, set a strong `FLOOD_APP_SECRET`, disable debug mode, add CSRF protection, file-size limits, and deployment hardening.

## Project structure
```text
flood_risk_prediction_system/
├── app.py
├── requirements.txt
├── README.md
├── flood_model.joblib      # generated at first run
├── flood_system.db         # generated at first run
├── data/
│   └── sample_flood_data.csv
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── predict.html
│   ├── result.html
│   ├── upload.html
│   ├── history.html
│   └── about.html
└── static/
    └── style.css
```
