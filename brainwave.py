"""
BrainWave: Binaural beats audio generator and information center
by Jacques Laroche

Key technologies:
- Flask, Python, HTML, CSS, Javascript

This file is intended to be:
- run directly with: `python brainwave.py` (dev)
- imported by a WSGI server as: `brainwave:app`

Important operational notes:
- ffmpeg must be installed in the runtime image/container for MP3/OGG/FLAC export.
- WAV export is intentionally disabled because WAV files are too large (WAV is only used as a temporary intermediate).

Output cleanup policy:
- Before each export: delete files older than 24 hours
- After each export: if output dir > 2GB OR file count > 100, delete oldest until under caps

Output filename policy:
BW_C{carrier}_B{beat}_{MMDDYY}-{SUF}.{ext}
  - carrier: rounded int (e.g., 200)
  - beat: up to 2 decimals, trim trailing zeros, '.' -> '-'
          (e.g., 8.12 -> 8-12, 12.40 -> 12-4, 8.00 -> 8)
  - Suffix: 3 chars from A–Z0–9 (collision-checked)
"""

import json # Used for loading the "reading list" (stored in JSON format)
import os
import re
import secrets # Used for file naming / file creation (unique suffix)
import shutil
import string # Used for file naming / file creation
import struct
import subprocess
import tempfile
import time
import wave
import logging # Used for logging errors
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple
import numpy as np # Used for audio rendering
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix


# -----------------------
# Utilities
# -----------------------
def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)

def ensure_dirs(
    static_dir: Path,
    docs_public_dir: Path,
    data_dir: Path,
    output_dir: Path,
    presets_dir: Path,
) -> None:
    static_dir.mkdir(parents=True, exist_ok=True)
    docs_public_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    presets_dir.mkdir(parents=True, exist_ok=True)

# -----------------------
# Logger Helper
# -----------------------
def setup_logging(logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("brainwave")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if app factory is called more than once
    if logger.handlers:
        return logger

    file_handler = RotatingFileHandler(
        logs_dir / "bw_errors.log",
        maxBytes=2 * 1024 * 1024,  # 2MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # Optional: also log to stdout for Docker-friendly logs
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


# -----------------------
# Output naming + cleanup
# -----------------------
_ALPHANUM = string.ascii_uppercase + string.digits

def _beat_part(beat: float) -> str:
    """Up to 2 decimals, trim trailing zeros, '.' -> '-'."""
    s = f"{beat:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", "-")

def make_output_filename_unique(
    output_dir: Path,
    carrier: float,
    beat: float,
    ext: str,
    when: Optional[datetime] = None,
) -> str:
    when = when or datetime.now()
    c = int(round(carrier))
    b = _beat_part(beat)
    date_part = when.strftime("%m%d%y")  # MMDDYY
    ext = (ext or "mp3").lower().strip().lstrip(".")

    while True:
        suffix = "".join(secrets.choice(_ALPHANUM) for _ in range(3))
        filename = f"BW_C{c}_B{b}_{date_part}-{suffix}.{ext}"
        if not (output_dir / filename).exists():
            return filename

def cleanup_output_dir(
    output_dir: Path,
    *,
    max_age_hours: float = 24.0,
    max_total_bytes: int = 2 * 1024**3,  # 2GB
    max_files: int = 100,
    protect_paths: Optional[Set[Path]] = None,
    allowed_exts: Optional[Set[str]] = None,
) -> None:
    """
    1) Delete files older than max_age_hours
    2) Enforce caps (size and count) by deleting oldest first
    """
    protect_paths = protect_paths or set()
    allowed_exts = allowed_exts or {".mp3", ".flac", ".ogg"}

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    now = time.time()
    max_age_s = max_age_hours * 3600.0

    def iter_candidates() -> Iterable[Tuple[Path, float, int]]:
        for p in output_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in allowed_exts:
                continue
            if p in protect_paths:
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            yield p, st.st_mtime, st.st_size

    # 1) Age-based delete
    for p, mtime, _sz in list(iter_candidates()):
        if (now - mtime) > max_age_s:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    # 2) Cap enforcement (size and number of files)
    files = list(iter_candidates())
    files.sort(key=lambda t: t[1])  # oldest first
    total = sum(sz for _p, _mt, sz in files)

    i = 0
    while (len(files) > max_files) or (total > max_total_bytes):
        if i >= len(files):
            break
        p, _mtime, sz = files[i]
        try:
            p.unlink(missing_ok=True)
            total -= sz
        except Exception:
            pass
        i += 1


# -----------------------
# Audio rendering
# -----------------------
def _float_to_pcm_bytes(stereo: np.ndarray, bit_depth: int) -> bytes:
    #stereo: shape (N, 2) float in [-1, 1]. Returns little-endian PCM bytes.
    stereo = np.clip(stereo, -1.0, 1.0)

    if bit_depth == 16:
        pcm = (stereo * 32767.0).astype(np.int16)
        return pcm.tobytes()

    if bit_depth == 24:
        pcm32 = (stereo * 8388607.0).astype(np.int32)  # 2^23 - 1
        out = bytearray()
        for sample in pcm32.reshape(-1):
            out.extend(struct.pack("<i", int(sample))[:3])  # 24-bit little-endian
        return bytes(out)

    raise ValueError("bit_depth must be 16 or 24")

def render_binaural_to_wav(
    path: Path,
    carrier: float,
    beat: float,
    duration_s: float,
    volume: float,
    sample_rate: int,
    bit_depth: int,
) -> None:
    n = int(sample_rate * duration_s)
    if n <= 0:
        raise ValueError("duration_s too small")

    t = np.linspace(0, duration_s, n, endpoint=False, dtype=np.float64)
    left = np.sin(2 * np.pi * (carrier - beat / 2.0) * t)
    right = np.sin(2 * np.pi * (carrier + beat / 2.0) * t)
    stereo = np.vstack((left, right)).T * float(volume)

    pcm_bytes = _float_to_pcm_bytes(stereo, bit_depth=bit_depth)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2 if bit_depth == 16 else 3)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)

def export_audio(
    output_dir: Path,
    *,
    carrier: float,
    beat: float,
    duration_s: float,
    volume: float,
    fmt: str,
    sample_rate: int,
    bit_depth: int,
) -> Tuple[str, str]:
    fmt = (fmt or "mp3").lower().strip().lstrip(".")
    if fmt not in {"mp3", "flac", "ogg"}:
        fmt = "mp3"  # WAV intentionally disallowed

    # Before each export perform a file cleanup
    cleanup_output_dir(output_dir, max_age_hours=24.0, max_total_bytes=2 * 1024**3, max_files=100)

    filename = make_output_filename_unique(output_dir, carrier, beat, fmt)
    out_path = output_dir / filename

    # Export audio as MP3, OGG, or FLAC
    ffmpeg = shutil.which("ffmpeg")
    # Check if ffmpeg is installed, throw an error if not
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for MP3/FLAC/OGG exports (not found on server).")

    with tempfile.TemporaryDirectory() as td:
        tmp_wav = Path(td) / "tmp.wav"
        render_binaural_to_wav(tmp_wav, carrier, beat, duration_s, volume, sample_rate, bit_depth)

        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(tmp_wav)]
        cmd += ["-ar", str(sample_rate)]

        if fmt == "mp3":
            cmd += ["-codec:a", "libmp3lame", "-q:a", "3", str(out_path)]
        elif fmt == "ogg":
            cmd += ["-codec:a", "libvorbis", "-q:a", "5", str(out_path)]
        elif fmt == "flac":
            cmd += ["-codec:a", "flac", str(out_path)]

        subprocess.run(cmd, check=True, capture_output=True, text=True)

    # After writing (protect newest file)
    cleanup_output_dir(
        output_dir,
        max_age_hours=24.0,
        max_total_bytes=2 * 1024**3,
        max_files=100,
        protect_paths={out_path},
    )

    return f"/static/output/{filename}", filename


# -----------------------------
# BrainWave Preset Files (.bwp)
# -----------------------------
_MAX_NAME = 20
_MAX_DESC = 200

# Make sure chosen .bwp filename is legal
def _safe_preset_filename(original: str) -> str:
    base = (original or "preset.bwp").strip()
    base = base.split("/")[-1].split("\\")[-1]
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", base)
    if not base.lower().endswith(".bwp"):
        base = base + ".bwp"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{base}"

# Create .bwp file
def parse_bwp_text(text: str) -> Dict[str, Any]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip() and not ln.strip().startswith("#")]
    kv: Dict[str, str] = {}
    for ln in lines:
        if "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        kv[k.strip().lower()] = v.strip()

    name = kv.get("name", "").strip()
    desc = kv.get("desc", "").strip()
    carrier_s = kv.get("carrier", None)
    beat_s = kv.get("beat", None)

    if not name:
        raise ValueError("Preset is missing 'name='.")
    if len(name) > _MAX_NAME:
        raise ValueError(f"Preset name is too long (max {_MAX_NAME} characters).")
    if len(desc) > _MAX_DESC:
        raise ValueError(f"Preset description is too long (max {_MAX_DESC} characters).")
    if carrier_s is None:
        raise ValueError("Preset is missing 'carrier='.")
    if beat_s is None:
        raise ValueError("Preset is missing 'beat='.")

    try:
        carrier = float(carrier_s)
        beat = float(beat_s)
    except ValueError:
        raise ValueError("carrier and beat must be numeric.")

    carrier = float(max(50.0, min(carrier, 2000.0)))
    beat = float(max(0.1, min(beat, 40.0)))

    return {"name": name, "desc": desc, "carrier": carrier, "beat": beat}


# -----------------------
# Reading list
# -----------------------
def load_reading_list(reading_list_path: Path) -> Dict[str, Any]:
    if not reading_list_path.exists():
        starter = {"public_domain": [], "copyrighted": []}
        reading_list_path.parent.mkdir(parents=True, exist_ok=True)
        reading_list_path.write_text(json.dumps(starter, indent=2), encoding="utf-8")

    data = json.loads(reading_list_path.read_text(encoding="utf-8"))
    data.setdefault("public_domain", [])
    data.setdefault("copyrighted", [])
    return data


# ---------
# Flask App
# ---------
def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")

    # env-driven config
    app.config["SECRET_KEY"] = _env("SECRET_KEY", "dev-only-change-me")
    app.config["MAX_CONTENT_LENGTH"] = int(_env("MAX_UPLOAD_BYTES", "65536"))

    # reverse proxy headers
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

    base_dir = Path(app.root_path)
    static_dir = base_dir / "static"
    output_dir = static_dir / "output"
    presets_dir = static_dir / "presets"
    docs_public_dir = static_dir / "docs" / "public_domain"
    data_dir = static_dir / "data"
    reading_list_path = data_dir / "reading_list.json"
    logs_dir = base_dir / "logs"
    logger = setup_logging(logs_dir)

    ensure_dirs(static_dir, docs_public_dir, data_dir, output_dir, presets_dir)

    @app.after_request
    def add_security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        resp.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=(), usb=(), payment=()")
        return resp

    # -----------------------
    # API
    # -----------------------
    @app.post("/api/generate")
    def api_generate():
        # Back-compat endpoint: generates an MP3 using default 44.1kHz / 16-bit.
        data = request.get_json(force=True) or {}
        carrier = float(data.get("carrier", 200.0))
        beat = float(data.get("beat", 8.0))
        duration = float(data.get("duration", 10.0))
        volume = float(data.get("volume", 0.5))

        duration = max(1.0, min(duration, 60.0))
        volume = max(0.0, min(volume, 1.0))

        url, filename = export_audio(
            output_dir,
            carrier=carrier,
            beat=beat,
            duration_s=duration,
            volume=volume,
            fmt="mp3",
            sample_rate=44100,
            bit_depth=16,
        )
        return jsonify({"ok": True, "url": url, "filename": filename})

    @app.post("/api/export")
    def api_export():
        data = request.get_json(force=True) or {}

        mode = (data.get("mode") or "export").lower().strip()
        fmt = (data.get("format") or "mp3").lower().strip()

        # Initial Carrier Frequency, Beat Frequency and Volume Level on app load
        carrier = float(data.get("carrier", 200.0))
        beat = float(data.get("beat", 8.0))
        volume = float(data.get("volume", 0.5))

        length_seconds = float(data.get("length_seconds", 10.0))
        sample_rate = int(data.get("sample_rate", 44100))
        bit_depth = int(data.get("bit_depth", 16))

        volume = max(0.0, min(volume, 1.0))

        if mode == "preview":
            fmt = "mp3"
            sample_rate = 44100
            bit_depth = 16
            length_seconds = max(10.0, min(length_seconds, 30.0))
        else:
            length_seconds = max(60.0, min(length_seconds, 3600.0))
            if sample_rate not in (44100, 48000):
                sample_rate = 44100
            if bit_depth not in (16, 24):
                bit_depth = 16
            if fmt not in ("mp3", "flac", "ogg"):
                fmt = "mp3"

        try:
            url, filename = export_audio(
                output_dir,
                carrier=carrier,
                beat=beat,
                duration_s=length_seconds,
                volume=volume,
                fmt=fmt,
                sample_rate=sample_rate,
                bit_depth=bit_depth,
            )

            return jsonify({"ok": True, "url": url, "filename": filename})

        except subprocess.CalledProcessError as e:
            incident = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + "".join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(3))

            logger.error(
                "INCIDENT %s | ffmpeg failed | carrier=%s beat=%s fmt=%s sr=%s bd=%s len_s=%s | cmd=%r | stderr=%s | stdout=%s",
                incident,
                carrier, beat, fmt, sample_rate, bit_depth, length_seconds,
                getattr(e, "cmd", None),
                (e.stderr or "").strip(),
                (e.stdout or "").strip(),
            )

            return jsonify({"ok": False, "error": f"ffmpeg failed (id: {incident}). Check logs/bw_errors.log"}), 500

        except Exception as e:
            incident = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + "".join(
                secrets.choice(string.ascii_uppercase + string.digits) for _ in range(3)
            )
            logger.exception("INCIDENT %s | unexpected export error: %s", incident, str(e))
            return jsonify({"ok": False, "error": f"Export failed (id: {incident}). Check logs/bw_errors.log"}), 500

    @app.post("/api/presets/upload")
    def api_presets_upload():
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "No file provided."}), 400

        f = request.files["file"]
        if not f or not f.filename:
            return jsonify({"ok": False, "error": "No file selected."}), 400

        filename = f.filename.strip()
        if not filename.lower().endswith(".bwp"):
            return jsonify({"ok": False, "error": "File must end with .bwp."}), 400

        data_bytes = f.read()
        if not data_bytes:
            return jsonify({"ok": False, "error": "Empty preset file."}), 400
        if len(data_bytes) > 64_000:
            return jsonify({"ok": False, "error": "Preset file is too large."}), 400

        try:
            text = data_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return jsonify({"ok": False, "error": "Preset must be UTF-8 text."}), 400

        try:
            preset = parse_bwp_text(text)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

        safe_name = _safe_preset_filename(filename)
        (presets_dir / safe_name).write_text(text, encoding="utf-8")

        return jsonify({"ok": True, "preset": preset, "saved_as": safe_name})

    # -----------------------
    # Pages / files
    # -----------------------
    @app.get("/reading_list.json")
    def library_json():
        return jsonify(load_reading_list(reading_list_path))

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/about")
    def about():
        return render_template("about.html")

    @app.get("/library")
    def library():
        reading = load_reading_list(reading_list_path)
        return render_template("library.html", reading=reading)

    @app.get("/docs/<path:filename>")
    def docs_public_domain(filename: str):
        if ".." in filename or filename.startswith("/"):
            abort(400)
        file_path = docs_public_dir / filename
        if not file_path.exists() or not file_path.is_file():
            abort(404)
        return send_from_directory(docs_public_dir, filename, as_attachment=False)

    @app.get("/favicon.ico")
    def favicon():
        return send_from_directory(
            str(Path(app.static_folder) / "img"),
            "favicon.ico",
            mimetype="image/vnd.microsoft.icon",
        )

    return app


# WSGI (Web Server Gateway Interface) entrypoint
app = create_app()


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5890"))
    debug = os.environ.get("DEBUG", "0").strip().lower() in ("1", "true", "yes", "on")
    app.run(host=host, port=port, debug=debug)