import sys
import os
import uuid

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from format.pipeline import run_customer_video_vs_reference_csv_pipeline
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from model import compute_similarity_from_csv, compute_similarity_from_video
from supabase_helper import supabase
from supabase import create_client, ClientOptions
from grade_snapshots import SNAPSHOT_WEIGHTS, SNAPSHOT_NAMES
from video_pose import video_to_world_landmarks_csv, video_to_reference_format_csv
from format.pipeline import run_video_coaching_pipeline
from werkzeug.utils import secure_filename

# Racket detection uses OpenCV. If OpenCV is missing, it wont crash the whole web app at startup.
try:
    from racket_detector import detect_racket
except Exception as import_error:
    print(f"WARNING: racket_detector could not be imported: {import_error}")

    def detect_racket(_file_path):
        return True, "Racket detector unavailable; skipped check."

import json
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# Flask sorts JSON keys alphabetically by default, which breaks any ordered
# dict (like grade_results['snapshots']) sent to the frontend via |tojson.
# Keep insertion order intact so the phase bars render in serve-timeline
# order (start_pose -> ... -> finish_pose) instead of alphabetically.
app.json.sort_keys = False  # Flask >= 2.3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY must be configured.")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CSV_EXTENSIONS = {".csv"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# ── PLAYERS ─────────────────────────────────────────────
# Each player has their own reference_files list, so /analyze and /upload
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
            os.path.join(BASE_DIR, "reference_players/Shelton_formatted.csv"),
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

def get_user_supabase():
    """Create a request-specific Supabase client for RLS-protected queries."""
    access_token = session.get("access_token")

    if not access_token:
        raise RuntimeError("No Supabase access token is available.")

    client = create_client(
        SUPABASE_URL,
        SUPABASE_ANON_KEY,
        options=ClientOptions(
            auto_refresh_token=False,
            persist_session=False,
        ),
    )
    client.postgrest.auth(access_token)
    return client


def save_session(
    db,
    user_id,
    filename,
    player_key,
    player_name,
    player_style,
    score,
    report_data=None,
):
    """Save one analysis result using the current user's RLS token."""
    response = (
        db.table("sessions")
        .insert(
            {
                "user_id": user_id,
                "filename": filename,
                "player_key": player_key,
                "player_name": player_name,
                "player_style": player_style,
                "score": score,
                "report_data": report_data,
            }
        )
        .execute()
    )
    return response.data


def get_user_sessions(db, user_id):
    """Load the current user's analysis history using an RLS-aware client."""
    response = (
        db.table("sessions")
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


@app.route("/analyze")
def analyze():
    if not login_required():
        return redirect(url_for("login"))

    player_key = request.args.get("player", DEFAULT_PLAYER)
    if player_key not in PLAYERS:
        player_key = DEFAULT_PLAYER
    selected_player = PLAYERS[player_key]

    return render_template(
        "analyze.html",
        user=session["user"],
        players=PLAYERS,
        selected_player_key=player_key,
        selected_player=selected_player,
    )


@app.route("/myprogress")
def myprogress():
    if not login_required():
        return redirect(url_for("login"))

    try:
        db = get_user_supabase()
        sessions = get_user_sessions(db, session["user_id"])
    except Exception as e:
        print(f"PROGRESS ERROR: {e}")
        flash("Your login session has expired. Please log in again.")
        session.clear()
        return redirect(url_for("login"))

    # Session numbers count up from the user's first-ever upload (oldest = 1),
    # so the number reflects their actual timeline regardless of how the
    # table itself is sorted (newest-first, below).
    sessions_by_age = sorted(sessions, key=lambda s: s.get("created_at") or "")
    session_numbers = {s.get("id"): i + 1 for i, s in enumerate(sessions_by_age)}
    for s in sessions:
        s["session_number"] = session_numbers.get(s.get("id"))

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


@app.route("/session/<session_id>")
def view_session(session_id):
    if not login_required():
        return redirect(url_for("login"))

    try:
        db = get_user_supabase()
        response = (
            db.table("sessions")
            .select("*")
            .eq("id", session_id)
            .eq("user_id", session["user_id"])  # ensure users can only view their own sessions
            .single()
            .execute()
        )
        row = response.data
    except Exception as e:
        print(f"VIEW SESSION ERROR: {e}")
        flash("Could not load that session.")
        return redirect(url_for("myprogress"))

    if not row:
        flash("Session not found.")
        return redirect(url_for("myprogress"))

    report = row.get("report_data") or {}
    player = {"name": row.get("player_name"), "style": row.get("player_style"), "tips": []}

    return render_template(
        "result.html",
        user=session["user"],
        filename=row.get("filename"),
        player=player,
        score=row.get("score"),
        plot_path=report.get("plot_path"),
        grade_results=report.get("grade_results"),
        coaching_report=report.get("coaching_report"),
        SNAPSHOT_WEIGHTS=SNAPSHOT_WEIGHTS,
        SNAPSHOT_NAMES=SNAPSHOT_NAMES,
    )
# processing helper
def make_json_safe(value):
    if isinstance(value, dict):
        return {
            key: make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [make_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    return value

# ── MAIN ANALYSIS ─────────────────────────────────────────

@app.route("/upload", methods=["POST"])
def upload():
    """
    Validate and save the upload, then redirect immediately to processing.html.

    The expensive analysis is intentionally performed later by /process-upload
    so the browser can display the processing page while the request runs.
    """
    if not login_required():
        return redirect(url_for("login"))

    file = request.files.get("media")

    if not file or not file.filename:
        flash("No file uploaded.")
        return redirect(url_for("analyze"))

    original_filename = secure_filename(file.filename)

    if not original_filename:
        flash("Invalid filename.")
        return redirect(url_for("analyze"))

    ext = os.path.splitext(original_filename)[1].lower()

    if ext not in CSV_EXTENSIONS and ext not in VIDEO_EXTENSIONS:
        flash(
            "Please upload a CSV or video file "
            "(.csv, .mp4, .mov, .avi, .mkv, .webm)."
        )
        return redirect(url_for("analyze"))

    player_key = request.form.get("player_key", DEFAULT_PLAYER)

    if player_key not in PLAYERS:
        player_key = DEFAULT_PLAYER

    # A unique server filename prevents users who upload files with the same
    # name from overwriting one another.
    stored_filename = f"{uuid.uuid4().hex}_{original_filename}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_filename)

    try:
        file.save(save_path)
        print(f"UPLOAD SAVED: {save_path}")
        print(f"UPLOAD SIZE: {os.path.getsize(save_path)} bytes")
    except Exception as e:
        print(f"FILE SAVE ERROR: {e}")

        import traceback
        traceback.print_exc()

        flash("The uploaded file could not be saved.")
        return redirect(url_for("analyze", player=player_key))

    # Flask's default session is stored in a signed browser cookie, so retain
    # only small strings here—not the video bytes or analysis result.
    session["pending_upload"] = {
        "path": save_path,
        "filename": original_filename,
        "extension": ext,
        "player_key": player_key,
    }

    return redirect(url_for("processing"))


@app.route("/processing")
def processing():
    """Display the loading page for the upload waiting to be analyzed."""
    if not login_required():
        return redirect(url_for("login"))

    pending = session.get("pending_upload")

    if not pending:
        flash("There is no upload waiting to be processed.")
        return redirect(url_for("analyze"))

    player_key = pending.get("player_key", DEFAULT_PLAYER)

    if player_key not in PLAYERS:
        player_key = DEFAULT_PLAYER

    player = PLAYERS[player_key]

    return render_template(
        "processing.html",
        user=session["user"],
        filename=pending.get("filename"),
        player_name=player["name"],
        player=player,
    )


@app.route("/process-upload", methods=["POST"])
def process_upload():
    """
    Run the long video/CSV analysis request started by processing.html.

    Returns JSON containing the page that the browser should open after the
    analysis succeeds or fails.
    """
    if not login_required():
        return {
            "success": False,
            "redirect": url_for("login"),
            "error": "Your login session has expired. Please log in again.",
        }, 401

    pending = session.get("pending_upload")

    if not pending:
        return {
            "success": False,
            "redirect": url_for("analyze"),
            "error": "No pending upload was found.",
        }, 400

    save_path = pending.get("path")
    filename = pending.get("filename")
    ext = pending.get("extension")
    player_key = pending.get("player_key", DEFAULT_PLAYER)

    if player_key not in PLAYERS:
        player_key = DEFAULT_PLAYER

    player = PLAYERS[player_key]

    if not save_path or not os.path.isfile(save_path):
        session.pop("pending_upload", None)
        return {
            "success": False,
            "redirect": url_for("analyze", player=player_key),
            "error": "The uploaded file could not be found on the server.",
        }, 404

    try:
        # ── RACKET RELEVANCE CHECK (video uploads only) ──────
        if ext in VIDEO_EXTENSIONS:
            print("STARTING RACKET CHECK")
            passed, details = detect_racket(save_path)

            if not passed:
                print(f"RACKET CHECK FAILED for {filename}: {details}")
                return {
                    "success": False,
                    "redirect": url_for("analyze", player=player_key),
                    "error": (
                        "We couldn't detect a tennis racket in your video. "
                        "Please upload a clear video of your serve."
                    ),
                }, 400

        # ── MAIN VIDEO / CSV PIPELINE ─────────────────────────
        if ext in VIDEO_EXTENSIONS:
            print("STARTING VIDEO PIPELINE")

            reference_path = player["reference_files"][0]

            if not os.path.exists(reference_path):
                raise FileNotFoundError(
                    f"Reference CSV was not found: {reference_path}"
                )

            pipeline_result = run_customer_video_vs_reference_csv_pipeline(
                save_path,
                reference_path,
            )

            print("VIDEO PIPELINE COMPLETE")

            grade_results = pipeline_result.snapshot_grade
            coaching_report = pipeline_result.coaching_report

            overall_score = grade_results.get("overall_score")
            if overall_score is None:
                raise ValueError(
                    "The analysis completed without a comparable overall score."
                )

            score = float(overall_score)
            plot_path = None

        elif ext in CSV_EXTENSIONS:
            score, avg_z, ref_mean, user_traj = compute_similarity_from_csv(
                save_path,
                player["reference_files"],
            )

            score = round(float(score), 1)
            plot_path = create_plot(user_traj, ref_mean)
            grade_results = None
            coaching_report = None

        else:
            raise ValueError("The pending upload has an unsupported file type.")

        coaching_report_data = (
            coaching_report.to_dict()
            if coaching_report and hasattr(coaching_report, "to_dict")
            else coaching_report
        )

        # ── SAVE THE COMPLETED REPORT TO SUPABASE ─────────────
        db = get_user_supabase()

        saved_rows = save_session(
            db=db,
            user_id=session["user_id"],
            filename=filename,
            player_key=player_key,
            player_name=player["name"],
            player_style=player["style"],
            score=score,
            report_data={
                "grade_results": make_json_safe(grade_results),
                "coaching_report": make_json_safe(coaching_report_data),
                "plot_path": plot_path,
            },
        )

        if not saved_rows or not saved_rows[0].get("id"):
            raise RuntimeError(
                "Supabase did not return the saved analysis session ID."
            )

        session_id = saved_rows[0]["id"]

        return {
            "success": True,
            "redirect": url_for(
                "view_session",
                session_id=session_id,
            ),
        }

    except Exception as e:
        print(f"UPLOAD PROCESSING ERROR: {e}")

        import traceback
        traceback.print_exc()

        return {
            "success": False,
            "redirect": url_for("analyze", player=player_key),
            "error": f"Error processing file: {e}",
        }, 500

    finally:
        try:
            if save_path and os.path.exists(save_path):
                os.remove(save_path)
        except Exception as cleanup_error:
            print(f"UPLOAD CLEANUP ERROR: {cleanup_error}")

        session.pop("pending_upload", None)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ── RUN ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5001)
