import cgi
import json
import mimetypes
import re
import shutil
import threading
import time
import uuid
import webbrowser
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from model_inference import get_predictor


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_INDEX = UPLOAD_DIR / "index.json"
HOST = "127.0.0.1"
PORT = 8000


def safe_filename(name):
    stem = Path(name).stem[:80] or "file"
    suffix = Path(name).suffix[:20]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "file"
    return f"{stem}{suffix}"


def size_label(size):
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024


def load_uploads():
    if not UPLOAD_INDEX.exists():
        return []
    return json.loads(UPLOAD_INDEX.read_text(encoding="utf-8"))


def save_uploads(items):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_INDEX.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def preview_for(path, content_type):
    if content_type.startswith("image/"):
        return "image", ""
    if content_type == "application/pdf":
        return "pdf", ""
    if content_type.startswith("text/") or path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
        try:
            return "text", path.read_text(encoding="utf-8", errors="replace")[:1800]
        except Exception:
            return "none", ""
    return "none", ""


class AgeAppHandler(SimpleHTTPRequestHandler):
    server_version = "KoreanAgeApp/1.0"

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.path = "/index.html"
        if self.path == "/api/uploads":
            self._send_json(200, {"files": load_uploads()})
            return
        return super().do_GET()

    def do_POST(self):
        if self.path == "/predict":
            self.handle_predict()
            return
        if self.path == "/upload":
            self.handle_upload()
            return
        self._send_json(404, {"error": "not_found"})

    def handle_predict(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))
            image = payload.get("image")
            if not image:
                self._send_json(400, {"error": "image field is required"})
                return
            already_cropped = bool(payload.get("already_cropped", False))
            apply_webcam_correction = bool(payload.get("apply_webcam_correction", True))
            self._send_json(
                200,
                get_predictor().predict(
                    image,
                    already_cropped=already_cropped,
                    apply_webcam_correction=apply_webcam_correction,
                ),
            )
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def handle_upload(self):
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type"),
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                },
            )
            file_item = form["file"] if "file" in form else None
            if file_item is None or not file_item.filename:
                self._send_json(400, {"error": "file is required"})
                return

            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            original_name = Path(file_item.filename).name
            stored_name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{safe_filename(original_name)}"
            stored_path = UPLOAD_DIR / stored_name
            with stored_path.open("wb") as out:
                shutil.copyfileobj(file_item.file, out)

            content_type = mimetypes.guess_type(original_name)[0] or "application/octet-stream"
            preview_type, preview_text = preview_for(stored_path, content_type)
            items = load_uploads()
            item = {
                "id": uuid.uuid4().hex,
                "title": form.getfirst("title", "").strip() or original_name,
                "description": form.getfirst("description", "").strip(),
                "original_name": original_name,
                "stored_name": stored_name,
                "download_url": f"/uploads/{stored_name}",
                "content_type": content_type,
                "preview_type": preview_type,
                "preview_text": preview_text,
                "size": stored_path.stat().st_size,
                "size_label": size_label(stored_path.stat().st_size),
                "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            items.insert(0, item)
            save_uploads(items)
            self._send_json(200, {"ok": True, "file": item})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def translate_path(self, path):
        path = unquote(path.split("?", 1)[0].split("#", 1)[0])
        relative = path.lstrip("/")
        resolved = (BASE_DIR / relative).resolve()
        if not str(resolved).startswith(str(BASE_DIR.resolve())):
            return str(BASE_DIR / "index.html")
        return str(resolved)

    def guess_type(self, path):
        if path.endswith(".js"):
            return "application/javascript"
        if path.endswith(".css"):
            return "text/css"
        return mimetypes.guess_type(path)[0] or "application/octet-stream"


def main():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if not UPLOAD_INDEX.exists():
        save_uploads([])
    url = f"http://{HOST}:{PORT}"
    print(f"Starting server: {url}")
    print(f"Project dir: {BASE_DIR}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    httpd = ThreadingHTTPServer((HOST, PORT), AgeAppHandler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
