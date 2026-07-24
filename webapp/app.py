from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from model import compute_similarity_from_csv, compute_similarity_from_video
from supabase_helper import supabase
from grading_pipeline import analyze_and_grade_video
from grade_snapshots import SNAPSHOT_WEIGHTS, SNAPSHOT_NAMES
from video_pose import video_to_world_landmarks_csv, video_to_reference_format_csv

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

app.json.sort_keys = False          # Flask >= 2.3


UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_EXTENSIONS = {".csv"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# ── PLAYERS ─────────────────────────────────────────────
# Each player has their own reference_files list, so /analyse and /upload
# stay in sync automatically off this one dict.
# Placeholder players (rybakina, sabalenka, roddick) currently point at
# Max's CSVs until real reference data is added — swap reference_files
# when ready, no other code needs to change.

PLAYERS = {
    "rybakina": {
        "name": "Rybakina",
        "style": "Power Serve",
        "avatar": "player6.png",
        "reference_files": [
            os.path.join(BASE_DIR, "reference_players/rybakina_formatted.csv")
        ],
        "tips": [
            "Reference data coming soon — currently using placeholder motion data.",
        ],
    },
    "sabalenka": {
        "name": "Sabalenka",
        "style": "Power Serve",
        "avatar": "player1.png",
        "reference_files": [
            os.path.join(BASE_DIR, "reference_players/sabalenka_formatted.csv")
        ],
        "tips": [
            "Reference data coming soon — currently using placeholder motion data.",
        ],
    },
    "roddick": {
        "name": "Roddick",
        "style": "Flat Serve",
        "avatar": "player5.png",
        "reference_files": [
            os.path.join(BASE_DIR, "reference_players/roddick_formatted.csv")
        ],
        "tips": [
            "Reference data coming soon — currently using placeholder motion data.",
        ],
    },
    "max": {
        "name": "B.Shelton",
        "style": "Kick Serve",
        "avatar": "player4.png",
        "reference_files": [
            os.path.join(BASE_DIR, "reference_players/shelton_formatted.csv"),

        ],
        "tips": [
            "Your serve is being compared to Max's recorded Vicon motion data.",
            "Focus on matching timing and trajectory.",
            "Differences in motion path will reduce your similarity score.",
            "Work on consistency across your swing.",
        ],
    },
}

DEFAULT_PLAYER = "max"


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


def restore_supabase_session():
    """Attach the current Flask user's Supabase token before RLS-protected queries."""
    access_token = session.get("access_token")
    refresh_token = session.get("refresh_token")

    if not access_token or not refresh_token:
        return

    try:
        supabase.auth.set_session(access_token, refresh_token)
    except Exception as e:
        print(f"SUPABASE SESSION RESTORE ERROR: {e}")


# ── GRAPH FUNCTION ─────────────────────────────────────────

def create_plot(user_traj, reference_traj):
    plt.figure()
    plt.plot(user_traj, label="Your serve")
    plt.plot(reference_traj, label="Reference")
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
            session["refresh_token"] = response.session.refresh_token

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

            # If email confirmation is enabled in Supabase, sign_up returns a user
            # but no active session. Do not treat that as logged in yet.
            if response.session is None:
                flash("Account created. Please check your email to confirm your account before logging in.")
                return redirect(url_for("login"))

            session["user"] = email
            session["user_id"] = response.user.id
            session["access_token"] = response.session.access_token
            session["refresh_token"] = response.session.refresh_token

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
    return render_template("index.html", user=session["user"], players=PLAYERS)


@app.route("/analyse")
def analyse():
    if not login_required():
        return redirect(url_for("login"))

    player_key = request.args.get("player", DEFAULT_PLAYER)
    if player_key not in PLAYERS:
        player_key = DEFAULT_PLAYER
    selected_player = PLAYERS[player_key]

    return render_template(
        "analyse.html",
        user=session["user"],
        players=PLAYERS,
        selected_player_key=player_key,
        selected_player=selected_player,
    )


@app.route("/myprogress")
def myprogress():
    if not login_required():
        return redirect(url_for("login"))

    restore_supabase_session()

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

    restore_supabase_session()

    file = request.files.get("media")

    if not file or file.filename == "":
        flash("No file uploaded.")
        return redirect(url_for("analyse"))

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    if ext not in CSV_EXTENSIONS and ext not in VIDEO_EXTENSIONS:
        flash("Please upload a CSV or video file (.csv, .mp4, .mov, .avi, .mkv, .webm).")
        return redirect(url_for("analyse"))

    player_key = request.form.get("player_key", DEFAULT_PLAYER)
    if player_key not in PLAYERS:
        player_key = DEFAULT_PLAYER
    player = PLAYERS[player_key]

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
            return redirect(url_for("analyse", player=player_key))

    # ── SCORING ───────────────────────────────────────────────
    # Video uploads go through the full snapshot-based grading pipeline:
    # raw MediaPipe landmarks -> format_data_mediapipe -> find_snapshots
    # -> grade_snapshots, compared against the standing reference serve
    # (see grading_pipeline.build_reference_serve() / REFERENCE_FORMATTED_CSV).
    #
    # CSV uploads still go through the older angle/DTW similarity path for
    # now -- they'd need their own run through format_data.py (the raw,
    # unmarked-Vicon-track version) + find_snapshots.py before they could
    # use grade_snapshots.py the same way a video does. Left as a follow-up.
    try:
        if ext in CSV_EXTENSIONS:
            score, avg_z, ref_mean, user_traj = compute_similarity_from_csv(
                save_path, player["reference_files"]
            )
            score = round(score, 1)
            plot_path = create_plot(user_traj, ref_mean)
            grade_results = None
        else:
            grade_results = analyze_and_grade_video(save_path, reference_csv_path=player["reference_files"][0])
            score = grade_results["overall_score"]
            plot_path = None

    except Exception as e:
        print(f"UPLOAD ERROR: {e}")
        import traceback

        traceback.print_exc()
        flash(f"Error processing file: {str(e)}")
        return redirect(url_for("analyse", player=player_key))

    try:
        save_session(
            user_id=session["user_id"],
            filename=filename,
            player_key=player_key,
            player_name=player["name"],
            player_style=player["style"],
            score=score,
        )
    except Exception as e:
        print(f"SAVE SESSION ERROR: {e}")
        flash("Analysis completed, but progress history could not be saved.")

    return render_template(
        "result.html",
        user=session["user"],
        filename=filename,
        player=player,
        score=score,
        plot_path=plot_path,
        grade_results=grade_results,
        SNAPSHOT_WEIGHTS=SNAPSHOT_WEIGHTS,
        SNAPSHOT_NAMES=SNAPSHOT_NAMES,

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