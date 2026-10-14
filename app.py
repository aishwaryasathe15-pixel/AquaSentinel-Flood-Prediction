import os, sqlite3, io
from datetime import datetime
from functools import wraps
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "flood_system.db")
DATA_PATH = os.path.join(BASE_DIR, "data", "sample_flood_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "flood_model.joblib")
FEATURES = ["rainfall_mm", "water_level_m", "humidity_pct", "temperature_c", "elevation_m", "previous_flood"]
RISK_ORDER = ["Low", "Medium", "High", "Extreme"]
app = Flask(__name__)
app.secret_key = os.environ.get("FLOOD_APP_SECRET", "change-this-secret-before-deployment")

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created_at TEXT NOT NULL)""")
        con.execute("""CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL, location TEXT, rainfall_mm REAL, water_level_m REAL,
            humidity_pct REAL, temperature_c REAL, elevation_m REAL, previous_flood INTEGER,
            risk TEXT NOT NULL, probability REAL NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id))""")

def generate_demo_data():
    rng = np.random.default_rng(42)
    n = 1200
    rainfall = rng.gamma(2.2, 28, n).clip(0, 250)
    water = (0.25 + rainfall*0.012 + rng.normal(0, 0.55, n)).clip(0.05, 8)
    humidity = rng.normal(73, 13, n).clip(25, 100)
    temp = rng.normal(28, 5, n).clip(8, 45)
    elevation = rng.gamma(2.0, 35, n).clip(0, 500)
    previous = rng.binomial(1, 0.25, n)
    score = (rainfall/75) + (water/2.3) + (humidity/100)*0.6 + previous*0.9 - elevation/300
    labels = np.where(score < 1.15, "Low", np.where(score < 2.0, "Medium", np.where(score < 3.0, "High", "Extreme")))
    frame = pd.DataFrame({"date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "location": rng.choice(["River Zone A","Lowland B","Urban Zone C","Village D"], n),
        "rainfall_mm": rainfall.round(2), "water_level_m": water.round(2),
        "humidity_pct": humidity.round(1), "temperature_c": temp.round(1),
        "elevation_m": elevation.round(1), "previous_flood": previous, "flood_risk": labels})
    frame.to_csv(DATA_PATH, index=False)
    return frame

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    aliases = {"rainfall":"rainfall_mm", "rainfall_mm":"rainfall_mm", "water_level":"water_level_m",
        "water_level_m":"water_level_m", "humidity":"humidity_pct", "humidity_pct":"humidity_pct",
        "temperature":"temperature_c", "temperature_c":"temperature_c", "elevation":"elevation_m",
        "elevation_m":"elevation_m", "previous_flood":"previous_flood", "previous_flood_occurrence":"previous_flood",
        "flood_risk":"flood_risk", "risk_level":"flood_risk", "flood":"flood_risk", "date":"date", "location":"location"}
    return df.rename(columns={c: aliases[c] for c in df.columns if c in aliases})

def risk_from_binary(v):
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "flood", "flooded"): return "High"
    if s in ("0", "false", "no", "no flood", "noflood"): return "Low"
    return str(v).strip().title()

def prepare_training_frame(df):
    df = normalize_columns(df)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Missing feature columns: " + ", ".join(missing))
    if "flood_risk" not in df.columns:
        raise ValueError("Training CSV must include a 'flood_risk' target column (Low/Medium/High/Extreme, or 0/1).")
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["flood_risk"] = df["flood_risk"].apply(risk_from_binary)
    df = df.dropna(subset=FEATURES + ["flood_risk"])
    df = df[df["flood_risk"].isin(RISK_ORDER)]
    if len(df) < 20:
        raise ValueError("At least 20 valid labelled rows are needed to train the model.")
    if df["flood_risk"].nunique() < 2:
        raise ValueError("Training data must contain at least two risk categories.")
    return df

def train_model(df):
    X, y = df[FEATURES], df["flood_risk"]
    # A stratified split is only safe when every class has at least 2 examples.
    counts = y.value_counts()
    stratify = y if counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    model = RandomForestClassifier(n_estimators=180, random_state=42, class_weight="balanced", min_samples_leaf=2)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    joblib.dump(model, MODEL_PATH)
    return model, accuracy, len(df)

def load_dataset():
    if not os.path.exists(DATA_PATH):
        return generate_demo_data()
    return pd.read_csv(DATA_PATH)

def get_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    frame = load_dataset()
    frame = prepare_training_frame(frame)
    model, _, _ = train_model(frame)
    return model

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("home.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if len(username) < 3 or "@" not in email or len(password) < 8:
            flash("Use a username of at least 3 characters, a valid email, and a password of at least 8 characters.", "danger")
            return render_template("register.html")
        try:
            with db() as con:
                con.execute("INSERT INTO users(username,email,password_hash,created_at) VALUES(?,?,?,?)",
                    (username, email, generate_password_hash(password), datetime.now().isoformat(timespec="seconds")))
            flash("Account created. Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("That username or email is already registered.", "danger")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        identity = request.form.get("identity", "").strip().lower()
        password = request.form.get("password", "")
        with db() as con:
            user = con.execute("SELECT * FROM users WHERE lower(username)=? OR lower(email)=?", (identity, identity)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))
        flash("Incorrect username/email or password.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))

@app.route("/dashboard")
@login_required
def dashboard():
    uploaded_path = os.path.join(BASE_DIR, "data", "uploaded_dataset.csv")
    frame = pd.read_csv(uploaded_path) if os.path.exists(uploaded_path) else load_dataset()
    frame = normalize_columns(frame)
    if "date" not in frame.columns:
        frame["date"] = np.arange(len(frame))
    frame = frame.tail(180)
    labels = frame["flood_risk"].astype(str).tolist() if "flood_risk" in frame.columns else []
    with db() as con:
        predictions = con.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT 8", (session["user_id"],)).fetchall()
        total = con.execute("SELECT COUNT(*) FROM predictions WHERE user_id=?", (session["user_id"],)).fetchone()[0]
    counts = {risk: labels.count(risk) for risk in RISK_ORDER}
    chart = {"dates": [str(x)[:10] for x in frame["date"].tolist()],
             "rainfall": pd.to_numeric(frame.get("rainfall_mm", pd.Series(dtype=float)), errors="coerce").fillna(0).round(2).tolist(),
             "water": pd.to_numeric(frame.get("water_level_m", pd.Series(dtype=float)), errors="coerce").fillna(0).round(2).tolist(),
             "risks": [counts[x] for x in RISK_ORDER], "riskLabels": RISK_ORDER}
    return render_template("dashboard.html", username=session.get("username"), predictions=predictions,
        total=total, chart=chart, counts=counts, data_rows=len(frame))

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        try:
            location = request.form.get("location", "").strip()[:100] or "Unspecified location"
            values = {}
            for feature in FEATURES:
                raw = request.form.get(feature, "").strip()
                values[feature] = int(float(raw)) if feature == "previous_flood" else float(raw)
            if not (0 <= values["rainfall_mm"] <= 1000 and 0 <= values["water_level_m"] <= 30 and
                    0 <= values["humidity_pct"] <= 100 and -10 <= values["temperature_c"] <= 60 and
                    0 <= values["elevation_m"] <= 9000 and values["previous_flood"] in (0,1)):
                raise ValueError("Please enter values within the displayed ranges.")
            model = get_model()
            X = pd.DataFrame([values], columns=FEATURES)
            risk = str(model.predict(X)[0])
            probs = model.predict_proba(X)[0]
            probability = float(max(probs))
            with db() as con:
                con.execute("""INSERT INTO predictions(user_id,created_at,location,rainfall_mm,water_level_m,humidity_pct,
                    temperature_c,elevation_m,previous_flood,risk,probability) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (session["user_id"], datetime.now().isoformat(timespec="seconds"), location,
                     values["rainfall_mm"], values["water_level_m"], values["humidity_pct"], values["temperature_c"],
                     values["elevation_m"], values["previous_flood"], risk, probability))
            return render_template("result.html", risk=risk, probability=probability, values=values, location=location)
        except Exception as e:
            flash(str(e), "danger")
    return render_template("predict.html")

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        file = request.files.get("dataset")
        if not file or not file.filename.lower().endswith(".csv"):
            flash("Please select a CSV file.", "danger")
            return redirect(url_for("upload"))
        try:
            incoming = pd.read_csv(file)
            normalized = normalize_columns(incoming)
            # Always keep a copy of the user's uploaded data for dashboard visualizations.
            upload_path = os.path.join(BASE_DIR, "data", "uploaded_dataset.csv")
            normalized.to_csv(upload_path, index=False)
            trained_message = "Dataset uploaded for visualization."
            if "flood_risk" in normalized.columns:
                cleaned = prepare_training_frame(normalized)
                _, accuracy, rows = train_model(cleaned)
                trained_message = f"Model retrained successfully using {rows} labelled rows. Hold-out accuracy: {accuracy:.1%}."
                # Use the uploaded file as the active dataset after successful training.
                cleaned.to_csv(DATA_PATH, index=False)
            flash(trained_message, "success")
            return redirect(url_for("dashboard"))
        except Exception as e:
            flash("Upload could not be processed: " + str(e), "danger")
    return render_template("upload.html")

@app.route("/history")
@login_required
def history():
    with db() as con:
        rows = con.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC", (session["user_id"],)).fetchall()
    return render_template("history.html", rows=rows)

@app.route("/sample-data")
@login_required
def sample_data():
    return send_file(DATA_PATH, as_attachment=True, download_name="sample_flood_data.csv")

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    init_db()
    if not os.path.exists(DATA_PATH):
        generate_demo_data()
    # Train the first model on the included demo data.
    if not os.path.exists(MODEL_PATH):
        train_model(prepare_training_frame(load_dataset()))
    app.run(debug=True)
