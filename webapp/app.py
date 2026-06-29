from flask import Flask, render_template, request, redirect, url_for, session, flash
from supabase_helper import supabase
from model import compute_similarity, load_csv, build_reference_model
import matplotlib.pyplot as plt
import os
import json
import hashlib
import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import datetime

app = Flask(__name__)
app.secret_key = "tennis_secret_key_change_in_production"

UPLOAD_FOLDER = "uploads"
USERS_FILE = "users.json"
DB_FILE = "tennisiq.db"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ IMPORTANT: FIXED PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REFERENCE_FILES = [
    os.path.join(BASE_DIR, "data/max_serves/max1.csv"),
    os.path.join(BASE_DIR, "data/max_serves/max2.csv"),
    os.path.join(BASE_DIR, "data/max_serves/max3.csv"),
    os.path.join(BASE_DIR, "data/max_serves/max4.csv"),
    os.path.join(BASE_DIR, "data/max_serves/max5.csv"),
]

# ── DATABASE ─────────────────────────────────────────────


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            filename TEXT NOT NULL,
            player_key TEXT NOT NULL,
            player_name TEXT NOT NULL,
            player_style TEXT NOT NULL,
            score REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """
    )
    conn.commit()
    conn.close()

# session of a user with a specific reference player and score, 
# along with the filename and timestamp. 
# This allows us to track the user's progress over time and provide- 
# feedback based on their performance compared to the reference player.
def save_session(user_id, filename, player_key, player_name, player_style, score):
    supabase.table("sessions").insert({
        "user_id": user_id,
        "filename": filename,
        "player_key": player_key,
        "player_name": player_name,
        "player_style": player_style,
        "score": score
    }).execute()


def get_user_sessions(user_id):
    response = (
        supabase.table("sessions")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )

    return response.data or []


init_db()

# ── AUTH ─────────────────────────────────────────────


def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# ── FEEDBACK ─────────────────────────────────────────────

PLAYER_FEEDBACK = {
    "max": {
        "name": "Max (Reference Player)",
        "style": "Model Serve",
        "tips": [
            "Your serve is being compared to Max's recorded Vicon motion data.",
            "Focus on matching timing and trajectory.",
            "Differences in motion path will reduce your similarity score.",
            "Work on consistency across your swing.",
        ],
    }
}

# ── GRAPH FUNCTION ─────────────────────────────────────────


def create_plot(file_path):
    df = pd.read_csv(file_path, skiprows=3, header=None)
    numeric = df.select_dtypes(include=[np.number])
    
    plt.figure()
    plt.plot(numeric.values)
    plot_filename = "plot.png"
    plot_path = os.path.join("static", plot_filename)
    plt.savefig(plot_path)
    plt.close()
    return plot_filename

# ── ROUTES ───────────────────────────────────────────────


@app.route("/")
def index():
    if "user" in session:
        return redirect(url_for("home"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["username"]
        password = request.form["password"]

        try:
            response = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })

            if response.user is None or response.session is None:
                flash("Invalid email or password")
                return render_template("login.html")

            session["user"] = email
            session["user_id"] = response.user.id
            session["access_token"] = response.session.access_token

            return redirect(url_for("home"))

        except Exception:
            flash("Invalid email or password")
            return render_template("login.html")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["username"]
        password = request.form["password"]
        confirm = request.form["confirm"]

        if password != confirm:
            flash("Passwords do not match")
            return render_template("register.html")

        try:
            response = supabase.auth.sign_up({
                "email": email,
                "password": password
            })

            if response.user is None:
                flash("Could not create account")
                return render_template("register.html")

            session["user"] = email
            session["user_id"] = response.user.id

            if response.session:
                session["access_token"] = response.session.access_token

            return redirect(url_for("home"))

        except Exception as e:
            flash(f"Registration failed: {e}")
            return render_template("register.html")

    return render_template("register.html")


@app.route("/home")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", user=session["user"])


@app.route("/analyse")
def analyse():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("analyse.html", user=session["user"])


@app.route("/myprogress")
def myprogress():
    if "user" not in session:
        return redirect(url_for("login"))

    sessions = get_user_sessions(session["user_id"])

    chart_sessions = list(reversed(sessions[:10]))
    chart_labels = [s["created_at"][5:10] for s in chart_sessions]
    chart_scores = [s["score"] for s in chart_sessions]

    avg_score = (
        round(sum(s["score"] for s in sessions) / len(sessions), 1) if sessions else 0
    )
    best_score = max((s["score"] for s in sessions), default=0)

    return render_template(
        "myprogress.html",
        user=session["user"],
        sessions=sessions,
        chart_labels=json.dumps(chart_labels),
        chart_scores=json.dumps(chart_scores),
        avg_score=avg_score,
        best_score=best_score,
        total_sessions=len(sessions),
    )


# ── MAIN ANALYSIS ─────────────────────────────────────────


@app.route("/upload", methods=["POST"])
def upload():
    if "user" not in session:
        return redirect(url_for("login"))

    file = request.files.get("media")

    if not file or file.filename == "":
        flash("No file uploaded.")
        return redirect(url_for("analyse"))

    if not file.filename.lower().endswith(".csv"):
        flash("Please upload a CSV file.")
        return redirect(url_for("analyse"))

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(save_path)

    try:
        score = compute_similarity(save_path, REFERENCE_FILES)
        score = round(score, 1)

        plot_path = create_plot(save_path)

    except Exception as e:
        print(f"UPLOAD ERROR: {e}")
        import traceback
        traceback.print_exc()
        flash(f"Error processing file: {str(e)}")
        return redirect(url_for("analyse"))

    feedback = PLAYER_FEEDBACK["max"]

    save_session(
        session["user_id"],
        file.filename,
        "max",
        feedback["name"],
        feedback["style"],
        score
    )

    return render_template(
        "result.html",
        user=session["user"],
        filename=file.filename,
        player=feedback,
        score=score,
        plot_path=plot_path,
    )


@app.route("/logout")
def logout():
    try:
        supabase.auth.sign_out()
    except Exception:
        pass

    session.clear()
    return redirect(url_for("login"))


# ── RUN ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5001)
