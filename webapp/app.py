from flask import Flask, render_template, request, redirect, url_for, session, flash
from model import compute_similarity_from_csv, compute_similarity_from_video
from supabase_helper import supabase

# Racket detection uses OpenCV. If OpenCV is missing, it wont crash the whole web app at startup.
try:
    from racket_detector import detect_racket
except Exception as import_error:
    print(f"WARNING: racket_detector could not be imported: {import_error}")

    def detect_racket(_file_path):
        return True, "Racket detector unavailable; skipped check."

import os
import json
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REFERENCE_FILES = [
    os.path.join(BASE_DIR, "data/max_serves/max1.csv"),
    os.path.join(BASE_DIR, "data/max_serves/max2.csv"),
    os.path.join(BASE_DIR, "data/max_serves/max3.csv"),
    os.path.join(BASE_DIR, "data/max_serves/max4.csv"),
    os.path.join(BASE_DIR, "data/max_serves/max5.csv"),
]

CSV_EXTENSIONS = {".csv"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# ── DATABASE / SUPABASE ─────────────────────────────────────

def save_session(user_id, filename, player_key, player_name, player_style, score):
    """Save one analysis result to the Supabase sessions table."""
    response = (
        supabase.table("sessions")
        .insert(
            {
                "user_id": user_id,
                "filename": filename,
                "player_key": player_key,
                "player_name": player_name,
                "player_style": player_style,
                "score": score,
            }
        )
        .execute()
    )
    return response.data


def get_user_sessions(user_id):
    """Load the current user's analysis history from Supabase."""
    response = (
        supabase.table("sessions")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return response.data or []


# ── AUTH HELPERS ─────────────────────────────────────────────

def validate_password(password):
    errors = []
    if len(password) < 8:
        errors.append("at least 8 characters")
    if not re.search(r"[A-Z]", password):
        errors.append("at least one uppercase letter")
    if not re.search(r"[a-z]", password):
        errors.append("at least one lowercase letter")
    if not re.search(r"\d", password):
        errors.append("at least one number")
    return errors


def login_required():
    return "user" in session and "user_id" in session


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

def create_plot(user_traj, reference_traj):
    plt.figure()
    plt.plot(user_traj, label="Your serve")
    plt.plot(reference_traj, label="Reference (Max)")
    plt.legend()
    plot_filename = "plot.png"
    static_dir = os.path.join(BASE_DIR, "static")
    os.makedirs(static_dir, exist_ok=True)
    plot_path = os.path.join(static_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    return plot_filename


# ── ROUTES ───────────────────────────────────────────────

@app.route("/")
def index():
    if login_required():
        return redirect(url_for("home"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["username"].strip()
        password = request.form["password"]

        try:
            response = supabase.auth.sign_in_with_password(
                {
                    "email": email,
                    "password": password,
                }
            )

            if response.user is None or response.session is None:
                flash("Invalid email or password")
                return render_template("login.html")

            session["user"] = email
            session["user_id"] = response.user.id
            session["access_token"] = response.session.access_token

            return redirect(url_for("home"))

        except Exception as e:
            print(f"LOGIN ERROR: {e}")
            flash("Invalid email or password")
            return render_template("login.html")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["username"].strip()
        password = request.form["password"]
        confirm = request.form["confirm"]

        if password != confirm:
            flash("Passwords do not match")
            return render_template("register.html")

        errors = validate_password(password)
        if errors:
            flash("Password must include: " + ", ".join(errors))
            return render_template("register.html")

        try:
            response = supabase.auth.sign_up(
                {
                    "email": email,
                    "password": password,
                }
            )

            if response.user is None:
                flash("Could not create account")
                return render_template("register.html")

            session["user"] = email
            session["user_id"] = response.user.id

            if response.session:
                session["access_token"] = response.session.access_token

            return redirect(url_for("home"))

        except Exception as e:
            print(f"REGISTER ERROR: {e}")
            flash(f"Registration failed: {e}")
            return render_template("register.html")

    return render_template("register.html")


@app.route("/home")
def home():
    if not login_required():
        return redirect(url_for("login"))
    return render_template("index.html", user=session["user"])


@app.route("/analyse")
def analyse():
    if not login_required():
        return redirect(url_for("login"))
    return render_template("analyse.html", user=session["user"])


@app.route("/myprogress")
def myprogress():
    if not login_required():
        return redirect(url_for("login"))

    try:
        sessions = get_user_sessions(session["user_id"])
    except Exception as e:
        print(f"PROGRESS ERROR: {e}")
        flash("Could not load progress history.")
        sessions = []

    chart_sessions = list(reversed(sessions[:10]))
    chart_labels = [(s.get("created_at") or "")[5:10] for s in chart_sessions]
    chart_scores = [s.get("score", 0) for s in chart_sessions]

    avg_score = (
        round(sum(float(s.get("score", 0)) for s in sessions) / len(sessions), 1)
        if sessions
        else 0
    )
    best_score = max((float(s.get("score", 0)) for s in sessions), default=0)

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
    if not login_required():
        return redirect(url_for("login"))

    file = request.files.get("media")

    if not file or file.filename == "":
        flash("No file uploaded.")
        return redirect(url_for("analyse"))

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    if ext not in CSV_EXTENSIONS and ext not in VIDEO_EXTENSIONS:
        flash("Please upload a CSV or video file (.csv, .mp4, .mov, .avi, .mkv, .webm).")
        return redirect(url_for("analyse"))

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # Racket relevance check for video uploads only.
    if ext in VIDEO_EXTENSIONS:
        passed, details = detect_racket(save_path)
        if not passed:
            os.remove(save_path)
            print(f"RACKET CHECK FAILED for {filename}: {details}")
            flash("We couldn't detect a tennis racket in your video. Please upload a video of your serve.")
            return redirect(url_for("analyse"))

    try:
        if ext in CSV_EXTENSIONS:
            score, avg_z, ref_mean, user_traj = compute_similarity_from_csv(
                save_path, REFERENCE_FILES
            )
        else:
            score, avg_z, ref_mean, user_traj = compute_similarity_from_video(
                save_path, REFERENCE_FILES
            )

        score = round(score, 1)
        plot_path = create_plot(user_traj, ref_mean)

    except Exception as e:
        print(f"UPLOAD ERROR: {e}")
        import traceback

        traceback.print_exc()
        flash(f"Error processing file: {str(e)}")
        return redirect(url_for("analyse"))

    feedback = PLAYER_FEEDBACK["max"]

    try:
        save_session(
            user_id=session["user_id"],
            filename=filename,
            player_key="max",
            player_name=feedback["name"],
            player_style=feedback["style"],
            score=score,
        )
    except Exception as e:
        print(f"SAVE SESSION ERROR: {e}")
        flash("Analysis completed, but progress history could not be saved.")

    return render_template(
        "result.html",
        user=session["user"],
        filename=filename,
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
from flask import Flask, render_template, request, redirect, url_for, session, flash
from model import compute_similarity_from_csv, compute_similarity_from_video
from supabase_helper import supabase
from racket_detector import detect_racket
import matplotlib.pyplot as plt
import os
import json
import hashlib
import sqlite3
import numpy as np
import matplotlib
import datetime
import re
matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

UPLOAD_FOLDER = "uploads"
USERS_FILE = "users.json"
DB_FILE = "tennisiq.db"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REFERENCE_FILES = [
    os.path.join(BASE_DIR, "data/max_serves/max1.csv"),
    os.path.join(BASE_DIR, "data/max_serves/max2.csv"),
    os.path.join(BASE_DIR, "data/max_serves/max3.csv"),
    os.path.join(BASE_DIR, "data/max_serves/max4.csv"),
    os.path.join(BASE_DIR, "data/max_serves/max5.csv"),
]

CSV_EXTENSIONS = {".csv"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

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


def save_session(username, filename, player_key, player_name, player_style, score):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO sessions (username, filename, player_key, player_name, player_style, score, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (
            username,
            filename,
            player_key,
            player_name,
            player_style,
            score,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        ),
    )
    conn.commit()
    conn.close()


def get_user_sessions(username):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        """
        SELECT * FROM sessions WHERE username = ?
        ORDER BY created_at DESC
    """,
        (username,),
    )
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]


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


def validate_password(password):
    errors = []
    if len(password) < 8:
        errors.append("at least 8 characters")
    if not re.search(r"[A-Z]", password):
        errors.append("at least one uppercase letter")
    if not re.search(r"[a-z]", password):
        errors.append("at least one lowercase letter")
    if not re.search(r"\d", password):
        errors.append("at least one number")
    return errors


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


def create_plot(user_traj, reference_traj):
    plt.figure()
    plt.plot(user_traj, label="Your serve")
    plt.plot(reference_traj, label="Reference (Max)")
    plt.legend()
    plot_filename = "plot.png"
    static_dir = os.path.join(BASE_DIR, "static")
    os.makedirs(static_dir, exist_ok=True)
    plot_path = os.path.join(static_dir, plot_filename)
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
        username = request.form["username"]
        password = request.form["password"]
        users = load_users()

        if username in users and users[username] == hash_password(password):
            session["user"] = username
            return redirect(url_for("home"))

        flash("Invalid username or password")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        confirm = request.form["confirm"]

        if password != confirm:
            flash("Passwords do not match")
            return render_template("register.html")

        errors = validate_password(password)
        if errors:
            flash("Password must include: " + ", ".join(errors))
            return render_template("register.html")

        users = load_users()

        if username in users:
            flash("Username already taken")
            return render_template("register.html")

        users[username] = hash_password(password)
        save_users(users)

        session["user"] = username
        return redirect(url_for("home"))

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

    sessions = get_user_sessions(session["user"])

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

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    if ext not in CSV_EXTENSIONS and ext not in VIDEO_EXTENSIONS:
        flash("Please upload a CSV or video file (.csv, .mp4, .mov, .avi, .mkv, .webm).")
        return redirect(url_for("analyse"))

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # ── RACKET RELEVANCE CHECK (video uploads only) ─────────
    # CSV files have no visual content, so this only applies to videos.
    # Rejects the upload before it ever reaches the expensive
    # MediaPipe pose-extraction step.
    if ext in VIDEO_EXTENSIONS:
        passed, details = detect_racket(save_path)
        if not passed:
            os.remove(save_path)
            print(f"RACKET CHECK FAILED for {filename}: {details}")
            flash("We couldn't detect a tennis racket in your video. Please upload a video of your serve.")
            return redirect(url_for("analyse"))

    try:
        if ext in CSV_EXTENSIONS:
            score, avg_z, ref_mean, user_traj = compute_similarity_from_csv(
                save_path, REFERENCE_FILES
            )
        else:
            score, avg_z, ref_mean, user_traj = compute_similarity_from_video(
                save_path, REFERENCE_FILES
            )

        score = round(score, 1)
        plot_path = create_plot(user_traj, ref_mean)

    except Exception as e:
        print(f"UPLOAD ERROR: {e}")
        import traceback
        traceback.print_exc()
        flash(f"Error processing file: {str(e)}")
        return redirect(url_for("analyse"))

    feedback = PLAYER_FEEDBACK["max"]

    save_session(
        username=session["user"],
        filename=filename,
        player_key="max",
        player_name=feedback["name"],
        player_style=feedback["style"],
        score=score,
    )

    return render_template(
        "result.html",
        user=session["user"],
        filename=filename,
        player=feedback,
        score=score,
        plot_path=plot_path,
    )


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ── RUN ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5001)
