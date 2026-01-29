import os
import json
import time
import datetime
import queue
import threading
from threading import Thread

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

import smtplib
from email.message import EmailMessage
from playsound import playsound
from retinaface import RetinaFace
from deepface import DeepFace


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_JSON = os.path.join(BASE_DIR, "gallery_db.json")
UNKNOWN_DIR = os.path.join(BASE_DIR, "unknown_captures")
ALARM_WAV_PATH = os.path.join(BASE_DIR, "Alarm.wav")
os.makedirs(UNKNOWN_DIR, exist_ok=True)

CAM_INDEX = 0
FRAME_WIDTH = 960
PROCESS_EVERY_N_FRAMES = 1

MODEL_NAME = "VGG-Face"
DEFAULT_THRESHOLD = 0.40
UNKNOWN_ALERT_SECONDS = 10

EMAIL_ADDRESS = os.getenv("ALERT_EMAIL", "")
EMAIL_PASSWORD = os.getenv("ALERT_EMAIL_PASS", "")
TO_EMAIL = os.getenv("ALERT_TO_EMAIL", "")


def play_alarm():
    try:
        playsound(ALARM_WAV_PATH)
    except Exception as e:
        print(f"[ALARM ERROR] {e}")


def send_email_alert(image_path: str):
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD or not TO_EMAIL:
        print("[EMAIL] Missing env vars ALERT_EMAIL / ALERT_EMAIL_PASS / ALERT_TO_EMAIL")
        return

    msg = EmailMessage()
    msg["Subject"] = "Alert! Unknown face detected"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = TO_EMAIL
    msg.set_content("An unknown person was detected. See attached image.")

    try:
        with open(image_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="image", subtype="png", filename=os.path.basename(image_path))
    except Exception as e:
        print(f"[EMAIL] attach error: {e}")
        return

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"[EMAIL SENT] {image_path}")
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def detect_faces_retinaface(bgr_img):
    det = RetinaFace.detect_faces(bgr_img)
    boxes = []
    if isinstance(det, dict):
        for info in det.values():
            x1, y1, x2, y2 = info["facial_area"]
            score = float(info.get("score", 0.0))
            boxes.append((x1, y1, x2, y2, score))
    boxes.sort(key=lambda x: x[4], reverse=True)
    return boxes


def cosine_distance(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a /= (np.linalg.norm(a) + 1e-12)
    b /= (np.linalg.norm(b) + 1e-12)
    return float(1.0 - np.dot(a, b))


def embed_face_vgg(face_bgr):
    rep = DeepFace.represent(
        img_path=face_bgr,
        model_name=MODEL_NAME,
        detector_backend="skip",
        enforce_detection=False,
        align=False,
    )[0]
    return np.array(rep["embedding"], dtype=np.float32)


def load_gallery_json(path):
    with open(path, "r", encoding="utf-8") as f:
        db = json.load(f)
    thr = float(db.get("threshold", DEFAULT_THRESHOLD))
    people = db.get("people", {})
    gallery = {n: [np.array(e, dtype=np.float32) for e in embs] for n, embs in people.items()}
    return gallery, thr


def identify(emb, gallery, threshold):
    best_name = "Unknown"
    best_dist = 1e9
    for name, embs in gallery.items():
        for e in embs:
            d = cosine_distance(emb, e)
            if d < best_dist:
                best_dist = d
                best_name = name
    return ("Unknown", best_dist) if best_dist > threshold else (best_name, best_dist)


class HomeSecurityGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Home Security System")
        self.root.geometry("980x640")
        self.root.minsize(900, 600)

        # Theme (design)
        self.bg = "#0b1220"
        self.panel = "#0f1b2d"
        self.text = "#e6eefc"
        self.muted = "#9fb0cc"
        self.good = "#37d67a"
        self.bad = "#ff4d4d"
        self.warn = "#ffb020"

        self.root.configure(bg=self.bg)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TLabel", background=self.bg, foreground=self.text, font=("Segoe UI", 11))
        style.configure("Title.TLabel", background=self.bg, foreground=self.text, font=("Segoe UI", 16, "bold"))
        style.configure("Muted.TLabel", background=self.bg, foreground=self.muted, font=("Segoe UI", 10))

        # Header
        header = tk.Frame(root, bg=self.bg)
        header.pack(fill="x", padx=18, pady=(16, 10))
        ttk.Label(header, text="Home Security System", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Live Camera • RetinaFace + VGG-Face • Alerts", style="Muted.TLabel").pack(anchor="w", pady=(6, 0))

        body = tk.Frame(root, bg=self.bg)
        body.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        self.card = tk.Frame(body, bg=self.panel, highlightthickness=1, highlightbackground="#1b2a44")
        self.card.pack(fill="both", expand=True)

        topbar = tk.Frame(self.card, bg=self.panel)
        topbar.pack(fill="x", padx=14, pady=(12, 10))

        self.status_dot = tk.Canvas(topbar, width=12, height=12, bg=self.panel, highlightthickness=0)
        self.status_dot.pack(side="left", padx=(0, 8))
        self.dot_id = self.status_dot.create_oval(2, 2, 10, 10, fill=self.muted, outline="")

        self.status_label = tk.Label(topbar, text="Starting…", bg=self.panel, fg=self.text, font=("Segoe UI", 12, "bold"))
        self.status_label.pack(side="left")

        self.right_info = tk.Label(topbar, text="", bg=self.panel, fg=self.muted, font=("Segoe UI", 10))
        self.right_info.pack(side="right")

        self.video_label = tk.Label(self.card, bg="#000000")
        self.video_label.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        # runtime
        self.running = True
        self.cap = None

        self.gallery = {}
        self.threshold = DEFAULT_THRESHOLD

        self.frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self.last_imgtk = None

        # unknown state
        self.unknown_detected = False
        self.unknown_start_time = None
        self.unknown_alerted = False

        # worker
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

        # UI refresh
        self._ui_tick()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _set_status(self, text: str, state: str = "neutral"):
        if state == "known":
            color = self.good
        elif state == "unknown":
            color = self.bad
        elif state == "warning":
            color = self.warn
        else:
            color = self.muted

        self.status_label.config(text=text)
        self.status_dot.itemconfigure(self.dot_id, fill=color)

    def _worker_loop(self):
        # load gallery
        try:
            self.root.after(0, lambda: self._set_status("Loading face DB from JSON…", "neutral"))
            self.gallery, self.threshold = load_gallery_json(DB_JSON)
            self.root.after(0, lambda: self._set_status(f"DB loaded: {len(self.gallery)} people • threshold={self.threshold:.2f}", "neutral"))
        except Exception as e:
            self.gallery = {}
            self.root.after(0, lambda err=str(e): self._set_status(f"DB error: {err}", "unknown"))

        # open camera
        self.cap = cv2.VideoCapture(CAM_INDEX)
        if not self.cap.isOpened():
            self.root.after(0, lambda: self._set_status("Camera error: cannot open webcam", "unknown"))
            return

        frame_i = 0
        fps_counter = 0
        fps_t0 = time.time()

        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue

            if FRAME_WIDTH:
                h0, w0 = frame.shape[:2]
                if w0 > FRAME_WIDTH:
                    scale = FRAME_WIDTH / float(w0)
                    frame = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)))

            annotated = frame.copy()
            h, w = frame.shape[:2]
            frame_i += 1

            any_known = False
            any_unknown = False

            if (frame_i % PROCESS_EVERY_N_FRAMES == 0) and self.gallery:
                try:
                    boxes = detect_faces_retinaface(frame)
                    for (x1, y1, x2, y2, _score) in boxes:
                        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
                        face = frame[y1:y2, x1:x2]
                        if face.size == 0:
                            continue

                        try:
                            emb = embed_face_vgg(face)
                            name, dist = identify(emb, self.gallery, self.threshold)
                        except Exception:
                            name, dist = "Unknown", 1.0

                        if name == "Unknown":
                            any_unknown = True
                            color = (0, 0, 255)
                        else:
                            any_known = True
                            color = (0, 255, 0)

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, f"{name} d={dist:.2f}", (x1, max(0, y1 - 10)),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
                except Exception as e:
                    self.root.after(0, lambda err=str(e): self._set_status(f"Recognition error: {err}", "unknown"))

            now = time.time()
            if any_unknown and not any_known:
                if not self.unknown_detected:
                    self.unknown_detected = True
                    self.unknown_start_time = now
                    self.unknown_alerted = False

                elapsed = now - (self.unknown_start_time or now)
                cv2.putText(annotated, f"Unknown Face! {elapsed:.1f}s", (10, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)

                self.root.after(0, lambda e=elapsed: self._set_status(
                    f"UNKNOWN detected • {e:.1f}s / {UNKNOWN_ALERT_SECONDS}s", "unknown"
                ))

                if (not self.unknown_alerted) and elapsed >= UNKNOWN_ALERT_SECONDS:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = os.path.join(UNKNOWN_DIR, f"Unknown_{ts}.png")
                    cv2.imwrite(filename, annotated)

                    Thread(target=play_alarm, daemon=True).start()
                    Thread(target=send_email_alert, args=(filename,), daemon=True).start()

                    self.unknown_alerted = True
                    self.root.after(0, lambda: self._set_status("ALERT sent (UNKNOWN)", "warning"))
            else:
                self.unknown_detected = False
                self.unknown_start_time = None
                self.unknown_alerted = False

                if not self.gallery:
                    self.root.after(0, lambda: self._set_status("Live Camera (DB not loaded)", "warning"))
                else:
                    self.root.after(0, lambda: self._set_status("Live Camera", "neutral"))

            # FPS info on right
            fps_counter += 1
            dt = time.time() - fps_t0
            if dt >= 1.0:
                fps = fps_counter / dt
                fps_counter = 0
                fps_t0 = time.time()
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.root.after(0, lambda f=fps, n=now_str: self.right_info.config(text=f"{n}   •   FPS {f:.1f}"))

            # push frame to UI
            try:
                if self.frame_queue.full():
                    _ = self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(annotated)
            except Exception:
                pass

        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

    def _ui_tick(self):
        try:
            frame = self.frame_queue.get_nowait()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            lw = max(1, self.video_label.winfo_width())
            lh = max(1, self.video_label.winfo_height())
            iw, ih = img.size
            scale = min(lw / iw, lh / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            if nw > 0 and nh > 0:
                img = img.resize((nw, nh), Image.LANCZOS)

            imgtk = ImageTk.PhotoImage(img)
            self.last_imgtk = imgtk
            self.video_label.config(image=imgtk)
        except queue.Empty:
            pass
        except Exception:
            pass

        self.root.after(15, self._ui_tick)

    def on_close(self):
        self.running = False
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    HomeSecurityGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()




