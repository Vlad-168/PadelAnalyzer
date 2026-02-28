"""
Padel Match Video Analyzer
--------------------------
Requirements:
    pip install opencv-contrib-python numpy openpyxl pillow
    pip install ultralytics supervision   # optional – YOLOv8 + ByteTrack

Run:
    python main.py
"""

from __future__ import annotations

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont, ImageTk
from tkinter import ttk, filedialog, messagebox
import sys
import threading
import json
import os
import subprocess
from datetime import timedelta, datetime


# ═══════════════════════ OPTIONAL YOLO IMPORT ═══════════════════════════ #

def _import_yolo_silent():
    """
    Import ultralytics + supervision while suppressing the harmless
    NNPACK warning that torch emits at the C++ level (fd=2).
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd   = os.dup(2)          # save real stderr (fd 2)
    os.dup2(devnull_fd, 2)          # redirect fd 2 → /dev/null
    try:
        from ultralytics import YOLO as _yolo
        import supervision as _sv
        return _yolo, _sv
    finally:
        os.dup2(saved_fd, 2)        # restore stderr
        os.close(devnull_fd)
        os.close(saved_fd)

try:
    _YOLO_CLS, sv = _import_yolo_silent()
    _YOLO_AVAILABLE = True
except (ImportError, Exception):
    _YOLO_AVAILABLE = False

from collections import deque
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.utils import get_column_letter


# ═══════════════════════════ UTILITIES ══════════════════════════════════ #

def _cv_close():
    """
    Reliably close all OpenCV windows.
    On macOS (Cocoa), destroyAllWindows() only marks windows for deletion;
    they disappear visually only after several cv2.waitKey(1) calls
    that flush the AppKit event loop.
    """
    cv2.destroyAllWindows()
    for _ in range(30):
        cv2.waitKey(1)


def create_tracker():
    """
    Create the best available single-object tracker.
    OpenCV 4.13+ moved CSRT/KCF – try all options in order of preference.
    """
    for f in (
        lambda: cv2.TrackerMIL_create(),          # OpenCV 4.5+  (no legacy)
        lambda: cv2.legacy.TrackerCSRT_create(),  # OpenCV < 4.13 contrib
        lambda: cv2.TrackerCSRT_create(),
        lambda: cv2.legacy.TrackerKCF_create(),
        lambda: cv2.TrackerKCF_create(),
    ):
        try:
            return f()
        except AttributeError:
            continue
    raise RuntimeError("No tracker found.\npip install opencv-contrib-python")


def put_text(img: np.ndarray, text: str, pos: tuple,
             color=(0, 220, 255), size: int = 22) -> np.ndarray:
    """
    Draw text with Pillow (supports non-Latin characters and Unicode).
    img – BGR numpy array (OpenCV). Returns a modified copy.
    """
    pil  = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    font = None
    candidates = [
        # macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    r, g, b = color[2], color[1], color[0]   # BGR → RGB
    draw.text(pos, text, font=font, fill=(r, g, b))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def fmt_time(sec: float) -> str:
    """Format seconds as M:SS string."""
    return f"{int(sec)//60}:{int(sec)%60:02d}"


def pick_player_on_frame(video_path: str, player_name: str,
                         already_bboxes: list | None = None,
                         parent=None) -> tuple | None:
    """
    Interactive player selection – pure Tkinter + PIL (no cv2.imshow).

    • Draws YOLO bounding boxes directly on a PIL image inside a Canvas.
    • Click inside a box → that detection is selected.
    • Click outside all boxes → 80×160 rectangle centered on cursor.
    • Buttons ← / → or A/D/SPACE keys to navigate frames.
    • ESC or 'Cancel' to abort.
    Returns (x, y, w, h) in original video coordinates, or None.
    """
    if already_bboxes is None:
        already_bboxes = []

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w0    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    nav_step   = max(1, int(fps * 2))
    candidates = list(range(0, min(total, int(fps * 60)), nav_step)) or [0]

    # ── Load YOLO model once ──────────────────────────────────────────────
    model = None
    if _YOLO_AVAILABLE:
        import io as _io
        _se, sys.stderr = sys.stderr, _io.StringIO()
        try:
            model = _YOLO_CLS("yolov8n.pt")
        finally:
            sys.stderr = _se

    det_cache: dict[int, list] = {}

    def get_dets(fn: int, frm: np.ndarray) -> list:
        if fn not in det_cache:
            if model:
                res = model(frm, classes=[0], conf=0.30, verbose=False)[0]
                det_cache[fn] = [tuple(map(int, b.xyxy[0].tolist()))
                                 for b in res.boxes]
            else:
                det_cache[fn] = []
        return det_cache[fn]

    # ── Display scale ─────────────────────────────────────────────────────
    scr_w = (parent.winfo_screenwidth()  if parent else 1280) - 80
    scr_h = (parent.winfo_screenheight() if parent else 800)  - 120
    scale = min(scr_w / max(w0, 1), scr_h / max(h0, 1), 1.0)
    dw, dh = max(1, int(w0 * scale)), max(1, int(h0 * scale))

    # ── Tkinter dialog ────────────────────────────────────────────────────
    root  = parent or tk._default_root
    dlg   = tk.Toplevel(root)
    dlg.title(f"Select player: {player_name}")
    dlg.resizable(False, False)
    dlg.grab_set()

    COLORS_PIL = ["#00FF00", "#00C8FF", "#FF6400", "#6400FF"]
    state      = {"idx": 0, "selected": None}
    _photo     = [None]    # prevent GC of PhotoImage
    _dets_disp = [[]]      # detections in display coordinates

    # ── Widgets ───────────────────────────────────────────────────────────
    info_var  = tk.StringVar(value=f"Select '{player_name}' – click on the player")
    frame_var = tk.StringVar(value="")

    tk.Label(dlg, textvariable=info_var,  font=("Helvetica", 11), pady=4).pack()
    tk.Label(dlg, textvariable=frame_var, font=("Helvetica", 9),  fg="#888").pack()

    canvas = tk.Canvas(dlg, width=dw, height=dh, cursor="crosshair", bg="black")
    canvas.pack()

    nav = tk.Frame(dlg); nav.pack(pady=6)
    tk.Button(nav, text="← Prev", command=lambda: navigate(-1)).pack(side="left", padx=4)
    tk.Button(nav, text="Next →", command=lambda: navigate(+1)).pack(side="left", padx=4)
    tk.Button(nav, text="✕ Cancel", command=dlg.destroy).pack(side="left", padx=4)

    # ── Frame rendering ───────────────────────────────────────────────────
    def render(fn: int, frame: np.ndarray):
        dets_orig = get_dets(fn, frame)
        disp_dets = [(int(x1*scale), int(y1*scale),
                      int(x2*scale), int(y2*scale))
                     for x1, y1, x2, y2 in dets_orig]
        _dets_disp[0] = disp_dets

        small = cv2.resize(frame, (dw, dh))
        pil   = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        draw  = ImageDraw.Draw(pil)

        # Already-selected players – grey boxes
        for bx, by, bw, bh in already_bboxes:
            sx1, sy1 = int(bx*scale), int(by*scale)
            sx2, sy2 = int((bx+bw)*scale), int((by+bh)*scale)
            draw.rectangle([sx1, sy1, sx2, sy2], outline="#787878", width=2)
            draw.text((sx1+4, sy1+4), "selected", fill="#A0A0A0")

        # YOLO detection boxes
        for di, (sx1, sy1, sx2, sy2) in enumerate(disp_dets):
            col = COLORS_PIL[di % len(COLORS_PIL)]
            draw.rectangle([sx1, sy1, sx2, sy2], outline=col, width=3)
            draw.text((sx1+6, sy1+6), f"#{di+1}", fill=col)

        n    = len(disp_dets)
        hint = "Click on the player" if n else "None found – navigate to another frame"
        info_var.set(f"{hint}  —  '{player_name}'  ({n} found)")
        frame_var.set(f"Frame {fn+1} / {total}   (← / → to navigate)")

        photo = ImageTk.PhotoImage(pil)
        canvas.create_image(0, 0, anchor="nw", image=photo)
        _photo[0] = photo   # keep reference to prevent GC

    # ── Frame loading ─────────────────────────────────────────────────────
    def load(idx: int):
        fn = candidates[idx % len(candidates)]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            return
        render(fn, frame)

    def navigate(delta: int):
        state["idx"] = max(0, state["idx"] + delta) % len(candidates)
        load(state["idx"])

    # ── Canvas click handler ──────────────────────────────────────────────
    def on_click(event):
        x, y = event.x, event.y
        for sx1, sy1, sx2, sy2 in _dets_disp[0]:
            if sx1 <= x <= sx2 and sy1 <= y <= sy2:
                ox1 = int(sx1 / scale); oy1 = int(sy1 / scale)
                ox2 = int(sx2 / scale); oy2 = int(sy2 / scale)
                state["selected"] = (ox1, oy1, ox2 - ox1, oy2 - oy1)
                dlg.destroy(); return
        # Click outside any box – bbox centered on cursor
        ox = int(x / scale); oy = int(y / scale)
        state["selected"] = (max(0, ox - 40), max(0, oy - 80), 80, 160)
        dlg.destroy()

    canvas.bind("<Button-1>", on_click)
    dlg.bind("<Right>",  lambda e: navigate(+1))
    dlg.bind("<Left>",   lambda e: navigate(-1))
    dlg.bind("<space>",  lambda e: navigate(+1))
    dlg.bind("d",        lambda e: navigate(+1))
    dlg.bind("a",        lambda e: navigate(-1))
    dlg.bind("<Escape>", lambda e: dlg.destroy())

    load(0)
    dlg.wait_window()
    cap.release()
    return state["selected"]


def is_inverted(ts: float, switch_times: list) -> bool:
    """Return True if the court sides are currently inverted for this timestamp."""
    return sum(1 for t in switch_times if ts >= t) % 2 == 1


def seg_intersect(p1, p2, p3, p4) -> bool:
    """Return True if line segments p1-p2 and p3-p4 intersect."""
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    d1, d2 = cross(p3, p4, p1), cross(p3, p4, p2)
    d3, d4 = cross(p1, p2, p3), cross(p1, p2, p4)
    return (
        ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and
        ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))
    )


# ═══════════════════════ COURT CALIBRATION ══════════════════════════════ #

class CourtCalibration:
    """Stores 4 court corners + net line in pixel coordinates."""

    def __init__(self, corners: list, net_p1: tuple, net_p2: tuple):
        self.corners = corners
        self.net_p1  = net_p1
        self.net_p2  = net_p2
        self._poly   = np.array(corners, dtype=np.float32)

    def in_court(self, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(self._poly, (float(x), float(y)), False) >= 0

    def crosses_net(self, p1, p2) -> bool:
        return seg_intersect(p1, p2, self.net_p1, self.net_p2)

    def net_y(self, x: float) -> float:
        x1, y1 = self.net_p1
        x2, y2 = self.net_p2
        if abs(x2 - x1) < 1:
            return (y1 + y2) / 2
        return y1 + (x - x1) / (x2 - x1) * (y2 - y1)

    def net_side(self, x: float, y: float) -> int:
        """Return 0 if above net, 1 if below net (in video Y-axis terms)."""
        return 0 if y < self.net_y(x) else 1

    def dist_to_boundary(self, x: float, y: float) -> float:
        """Return signed distance from point to court boundary polygon (>0 = inside)."""
        return cv2.pointPolygonTest(self._poly, (float(x), float(y)), True)


def calibrate_court(video_path: str, parent=None) -> CourtCalibration | None:
    """
    Court calibration dialog – pure Tkinter + PIL (no cv2.imshow).

    The user clicks 6 points in a Toplevel window:
      1-4: court corners (clockwise from top-left)
      5-6: left and right ends of the net
    ESC or 'Cancel' button aborts.
    After the 6th click the window closes automatically after 700 ms.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    h0, w0 = frame.shape[:2]

    # ── Display scale ─────────────────────────────────────────────────────
    root_win = parent or tk._default_root
    scr_w = (parent.winfo_screenwidth()  if parent else 1280) - 80
    scr_h = (parent.winfo_screenheight() if parent else 800)  - 140
    scale = min(scr_w / max(w0, 1), scr_h / max(h0, 1), 1.0)
    dw, dh = max(1, int(w0 * scale)), max(1, int(h0 * scale))

    labels = [
        "1/6  Top-LEFT corner of court",
        "2/6  Top-RIGHT corner of court",
        "3/6  Bottom-RIGHT corner of court",
        "4/6  Bottom-LEFT corner of court",
        "5/6  LEFT end of net",
        "6/6  RIGHT end of net",
    ]

    pts_orig: list = []          # points in original video coordinates
    state          = {"result": None}
    _photo         = [None]      # prevent GC of PhotoImage

    # ── Tkinter dialog ────────────────────────────────────────────────────
    dlg = tk.Toplevel(root_win)
    dlg.title("Calibration – click points in order  |  ESC = cancel")
    dlg.resizable(False, False)
    dlg.grab_set()

    info_var = tk.StringVar(value=labels[0])
    tk.Label(dlg, textvariable=info_var,
             font=("Helvetica", 11), pady=6).pack()

    canvas = tk.Canvas(dlg, width=dw, height=dh, cursor="crosshair", bg="black")
    canvas.pack()

    tk.Button(dlg, text="✕ Cancel", command=dlg.destroy,
              font=("Helvetica", 10)).pack(pady=6)

    # Base PIL frame (scaled, without any annotations)
    frame_small = cv2.resize(frame, (dw, dh))
    base_pil = Image.fromarray(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))

    # ── Redraw current state ──────────────────────────────────────────────
    def redraw(done: bool = False):
        pil  = base_pil.copy()
        draw = ImageDraw.Draw(pil)

        disp = [(int(ox * scale), int(oy * scale)) for ox, oy in pts_orig]
        n    = len(disp)

        # Court outline
        if n >= 2:
            for i in range(1, min(n, 4)):
                draw.line([disp[i-1], disp[i]], fill="#00FF00", width=2)
        if n >= 4:
            draw.line([disp[3], disp[0]], fill="#00FF00", width=2)
        # Net line
        if n == 6:
            draw.line([disp[4], disp[5]], fill="#FF4040", width=3)

        # Numbered circles
        for i, (px, py) in enumerate(disp):
            r   = 7
            col = "#FF4040" if i >= 4 else "#00FF00"
            draw.ellipse([px - r, py - r, px + r, py + r],
                         fill=col, outline=col)
            draw.text((px + 10, py - 12), str(i + 1), fill=col)

        # Status text
        if done:
            draw.text((20, 14), "✓ All points set – closing…", fill="#00FF00")
        else:
            draw.text((20, 14), labels[min(n, 5)], fill="#00DCFF")

        photo = ImageTk.PhotoImage(pil)
        canvas.create_image(0, 0, anchor="nw", image=photo)
        _photo[0] = photo   # keep reference

    # ── Click handler ─────────────────────────────────────────────────────
    def on_click(event):
        if len(pts_orig) >= 6:
            return
        ox = int(event.x / scale)
        oy = int(event.y / scale)
        pts_orig.append((ox, oy))
        n = len(pts_orig)
        if n < 6:
            info_var.set(labels[n])
            redraw()
        else:
            info_var.set("✓ All points selected!")
            state["result"] = CourtCalibration(
                pts_orig[:4], pts_orig[4], pts_orig[5])
            redraw(done=True)
            dlg.after(700, dlg.destroy)

    canvas.bind("<Button-1>", on_click)
    dlg.bind("<Escape>", lambda e: dlg.destroy())

    redraw()
    dlg.wait_window()
    return state["result"]


# ══════════════════════════ BALL DETECTOR ═══════════════════════════════ #

class BallDetector:
    """
    Detects the padel ball using frame-differencing + HSV filtering.

    Strategy (inspired by TrackNet / tennis-ball CV research):
    ───────────────────────────────────────────────────────────
    1. FRAME DIFFERENCE:
       The ball moves fast between frames. Computing abs-diff of consecutive
       greyscale frames isolates moving objects. Combined with an HSV mask
       for yellow-green colour, this gives near-zero false positives from
       static clothing or court markings.

    2. TINY BLOB FOCUS:
       A padel ball at typical camera distance appears as a 5-30 px² blob.
       Size constraints are deliberately tight to exclude players' clothing
       and other large yellow/green objects.

    3. KALMAN FILTER (4D: x, y, vx, vy):
       After the first reliable detection the Kalman filter predicts future
       ball position. Subsequent candidates outside GATE_PX are rejected.
       This stabilises tracking through brief detection gaps.

    4. MOG2 FALLBACK:
       When no frame-diff candidate is found (ball stationary), the
       detector falls back to the HSV+MOG2 mask used in earlier versions.

    5. MULTI-STEP MISS TOLERANCE:
       The Kalman filter keeps predicting for up to MISS_MAX frames before
       resetting, bridging brief occlusions.
    """

    # HSV ranges for yellow-green padel ball (indoor LED lighting)
    HSV_RANGES = [
        (np.array([18, 50,  80]),  np.array([55, 255, 255])),  # tight yellow-green
        (np.array([14, 35,  60]),  np.array([62, 255, 255])),  # wider  (dim corners)
    ]
    # Size constraints – very tight to exclude player clothing
    MIN_AREA  = 4      # px²  (ball at far end ≈ 5 px²)
    MAX_AREA  = 800    # px²  (ball close to camera ≈ 300-500 px²)
    MIN_CIRC  = 0.30   # circularity 4πA/P²
    GATE_PX   = 500    # Kalman gating (px): ball reversal at 100km/h = ~274px/frame
    MISS_MAX  = 18     # frames before Kalman reset (longer = survives fast serves)
    DIFF_THR  = 18     # greyscale diff threshold for moving-object mask
    # Court polygon filter: reject ball candidates > BALL_COURT_MARGIN px outside court.
    # Prevents frame-edge false positives while allowing legitimate "out" balls.
    BALL_COURT_MARGIN = -60   # px  (allow up to 60 px outside court boundary)

    def __init__(self, court_poly: np.ndarray | None = None):
        self._prev_gray: np.ndarray | None = None
        self._prev_pos:  tuple | None = None
        self._kf         = self._init_kf()
        self._kf_active  = False
        self._miss_cnt   = 0
        self._court_poly = court_poly   # optional (N,2) float32 polygon

    @staticmethod
    def _init_kf():
        """4-state Kalman filter: state=(x, y, vx, vy), measurement=(x, y)."""
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.05
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.5
        kf.errorCovPost        = np.eye(4, dtype=np.float32) * 500.0
        return kf

    def detect(self, frame: np.ndarray,
               fgmask: np.ndarray | None = None) -> tuple | None:
        """
        Detect ball in frame.
        Returns (cx, cy) in pixel coordinates, or None if not found.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ── Build colour mask ─────────────────────────────────────────────
        colour_mask = np.zeros(gray.shape, dtype=np.uint8)
        for lo, hi in self.HSV_RANGES:
            colour_mask = cv2.bitwise_or(colour_mask,
                                         cv2.inRange(hsv, lo, hi))

        # ── Build motion mask (frame difference) ──────────────────────────
        if self._prev_gray is not None:
            diff = cv2.absdiff(gray, self._prev_gray)
            _, motion_mask = cv2.threshold(diff, self.DIFF_THR, 255,
                                           cv2.THRESH_BINARY)
            motion_mask = cv2.dilate(motion_mask,
                                     np.ones((3, 3), np.uint8), iterations=1)
            combined = cv2.bitwise_and(colour_mask, motion_mask)
        else:
            # First frame: use colour + MOG2 mask only
            combined = colour_mask
            if fgmask is not None:
                combined = cv2.bitwise_and(combined, fgmask)

        self._prev_gray = gray

        # Minimal morphology to remove isolated noise pixels
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,
                                    np.ones((2, 2), np.uint8))

        # ── Kalman prediction ─────────────────────────────────────────────
        gate_center = None
        if self._kf_active:
            pred        = self._kf.predict()
            gate_center = (float(pred[0]), float(pred[1]))

        best_ball  = None
        best_score = 0.0

        cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if not (self.MIN_AREA <= area <= self.MAX_AREA):
                continue
            perim = cv2.arcLength(c, True)
            if perim < 1:
                continue
            circ = 4 * np.pi * area / (perim * perim)
            if circ < self.MIN_CIRC:
                continue
            M = cv2.moments(c)
            if M["m00"] < 1:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            # Court polygon filter: reject candidates far outside court.
            # Eliminates false positives at frame edges (e.g. spectators, reflections).
            if self._court_poly is not None:
                court_dist = cv2.pointPolygonTest(
                    self._court_poly, (cx, cy), True)
                if court_dist < self.BALL_COURT_MARGIN:
                    continue

            # Kalman gating
            if gate_center is not None:
                gate_dist = np.hypot(cx - gate_center[0], cy - gate_center[1])
                if gate_dist > self.GATE_PX:
                    continue
                prox = max(0.0, 1.0 - gate_dist / self.GATE_PX)
            elif self._prev_pos:
                d    = np.hypot(cx - self._prev_pos[0], cy - self._prev_pos[1])
                prox = max(0.0, 1.0 - d / 300.0)
            else:
                prox = 0.0

            # Score = circularity weight + proximity weight
            score = circ * 0.55 + prox * 0.45
            if score > best_score:
                best_score = score
                best_ball  = (cx, cy)

        # ── Fallback: MOG2 mask when no frame-diff candidate found ─────────
        if best_ball is None and fgmask is not None and gate_center is not None:
            fg_combined = cv2.bitwise_and(colour_mask, fgmask)
            fg_combined = cv2.morphologyEx(fg_combined, cv2.MORPH_OPEN,
                                           np.ones((2, 2), np.uint8))
            cnts2, _ = cv2.findContours(fg_combined, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts2:
                area = cv2.contourArea(c)
                if not (self.MIN_AREA <= area <= self.MAX_AREA):
                    continue
                perim = cv2.arcLength(c, True)
                if perim < 1: continue
                circ = 4 * np.pi * area / (perim * perim)
                if circ < self.MIN_CIRC: continue
                M = cv2.moments(c)
                if M["m00"] < 1: continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                if self._court_poly is not None:
                    court_dist = cv2.pointPolygonTest(
                        self._court_poly, (cx, cy), True)
                    if court_dist < self.BALL_COURT_MARGIN:
                        continue
                gate_dist = np.hypot(cx - gate_center[0], cy - gate_center[1])
                if gate_dist > self.GATE_PX: continue
                prox  = max(0.0, 1.0 - gate_dist / self.GATE_PX)
                score = circ * 0.5 + prox * 0.5
                if score > best_score:
                    best_score = score
                    best_ball  = (cx, cy)

        # ── Kalman update ─────────────────────────────────────────────────
        if best_ball is not None:
            meas = np.array([[np.float32(best_ball[0])],
                             [np.float32(best_ball[1])]])
            if not self._kf_active:
                self._kf.statePost = np.array(
                    [[best_ball[0]], [best_ball[1]], [0.0], [0.0]], np.float32)
                self._kf_active = True
            self._kf.correct(meas)
            self._miss_cnt = 0
        elif self._kf_active:
            self._miss_cnt += 1
            if self._miss_cnt > self.MISS_MAX:
                self._kf_active = False
                self._miss_cnt  = 0

        self._prev_pos = best_ball
        return best_ball


# ═══════════════════ YOLO + BYTETRACK PLAYER TRACKER ════════════════════ #

class YOLOByteTracker:
    """
    Multi-object player tracker: YOLOv8n (detector) + ByteTrack (MOT).

    How it works:
    ─────────────
    1. On initialisation YOLO detects players in the first frame.
       Each initial bbox (user ROI) is matched to the nearest detection
       → a track_id ↔ pid mapping is established.

    2. On each frame YOLO re-detects all players.
       ByteTrack assigns stable track_ids across time.
       Returns {pid: (px_pixels, py_pixels)} for each tracked player.

    Advantages over CSRT:
    ──────────────────────
    • No drift accumulation – each frame has a fresh detection
    • ByteTrack handles occlusions and fast movements
    • Auto-recovery: lost player is found again automatically
    • ID swaps between players are impossible
    """

    CONF         = 0.18   # lower confidence to catch small far-end players
    BUFFER       = 60     # frames before a lost track is deleted
    FPS          = 12     # matches VideoAnalyzer step (~12 fps)
    MAX_MATCH_PX = 350    # max pixel distance for initial / late matching

    def __init__(self, initial_bboxes: list, first_frame: np.ndarray,
                 court_poly: np.ndarray | None = None, model=None):
        """
        court_poly – optional (N,2) float32 polygon to restrict player
                     detections to the court area, ignoring spectators.
        model      – optional pre-loaded YOLO model instance to avoid
                     reloading from disk on every initialisation.
        """
        if model is not None:
            self._model = model
        else:
            # Load model, suppressing harmless NNPACK warning
            import io
            _old_stderr, sys.stderr = sys.stderr, io.StringIO()
            try:
                self._model = _YOLO_CLS("yolov8n.pt")
            finally:
                sys.stderr = _old_stderr

        try:
            self._tracker = sv.ByteTrack(
                lost_track_buffer=self.BUFFER,
                frame_rate=self.FPS,
            )
        except TypeError:
            # supervision < 0.21 does not have frame_rate parameter
            self._tracker = sv.ByteTrack(lost_track_buffer=self.BUFFER)

        # Inflate court polygon by 80 px so players at the edges are included
        self._court_poly = court_poly

        self._tid_to_pid: dict[int, int] = {}
        self._pid_to_tid: dict[int, int] = {}
        # All players start as 'unmatched'
        self._initial_bboxes: list = list(initial_bboxes)  # saved for re-match fallback
        self._unmatched: dict[int, tuple] = {
            pid: bbox for pid, bbox in enumerate(initial_bboxes)}
        # Last known screen-pixel position for each PID (for re-matching after ID swap)
        self._last_pos: dict[int, tuple] = {}
        self._match_initial(first_frame)

    def _in_court(self, cx: float, cy: float, margin: float = 200.0) -> bool:
        """
        True if (cx, cy) is inside the court polygon plus a margin.
        Margin 200 px keeps near-camera players (body centre slightly inside court)
        while reliably excluding spectators (~300+ px outside the polygon).
        """
        if self._court_poly is None:
            return True
        dist = cv2.pointPolygonTest(self._court_poly, (float(cx), float(cy)), True)
        return dist >= -margin

    # ── Detection ────────────────────────────────────────────────────────

    def _detect(self, frame: np.ndarray) -> "sv.Detections":
        results = self._model(frame, classes=[0], conf=self.CONF,
                              verbose=False)[0]
        dets = sv.Detections.from_ultralytics(results)
        return self._tracker.update_with_detections(dets)

    # ── Match unmatched players to detections ─────────────────────────────

    def _try_match(self, dets: "sv.Detections"):
        """
        Match unmatched players to detections using minimum-distance assignment.

        Sorts ALL (pid, tid, distance) candidates globally so the closest
        pairs are assigned first — avoids the greedy ordering bug where an
        earlier pid 'steals' a detection that was closest to a later pid.
        """
        if not self._unmatched or len(dets) == 0 or dets.tracker_id is None:
            return

        # Build all valid (distance, pid, tid) candidate triples
        candidates: list = []
        for pid, (bx, by, bw, bh) in self._unmatched.items():
            icx = bx + bw / 2
            icy = by + bh / 2
            for i in range(len(dets)):
                tid = dets.tracker_id[i]
                if tid is None:
                    continue
                tid = int(tid)
                if tid in self._tid_to_pid:
                    continue   # already assigned
                x1, y1, x2, y2 = dets.xyxy[i]
                dcx, dcy = (x1 + x2) / 2, (y1 + y2) / 2
                if not self._in_court(dcx, dcy):
                    continue
                d = np.hypot(dcx - icx, dcy - icy)
                if d < self.MAX_MATCH_PX:
                    candidates.append((d, pid, tid))

        # Sort by distance – assign shortest pairs first (globally optimal greedy)
        candidates.sort()
        assigned_pids: set = set()
        assigned_tids: set = set()
        for d, pid, tid in candidates:
            if pid in assigned_pids or tid in assigned_tids:
                continue
            if pid not in self._unmatched:
                continue   # already matched in an earlier iteration
            self._tid_to_pid[tid] = pid
            self._pid_to_tid[pid] = tid
            del self._unmatched[pid]
            assigned_pids.add(pid)
            assigned_tids.add(tid)

    def _match_initial(self, frame: np.ndarray):
        dets = self._detect(frame)
        self._try_match(dets)

    # ── Per-frame update ──────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> dict:
        """
        Returns {pid: (px, py)} – centre of bbox in pixels for each
        tracked player.  Automatically matches players that could not be
        identified in the first frame (late matching).

        Also handles ByteTrack ID-swap recovery: when ByteTrack assigns a
        new tracker_id to a previously-tracked player (e.g. after a brief
        occlusion), the stale PID↔tid mapping is removed and the player is
        re-matched to the new detection using their last known position.
        This prevents persistent 'ghost' frames where no position is returned
        for a well-visible player.
        """
        dets = self._detect(frame)
        positions: dict[int, tuple] = {}

        if len(dets) == 0 or dets.tracker_id is None:
            return positions

        current_tids = {int(tid) for tid in dets.tracker_id if tid is not None}

        # Re-match PIDs whose ByteTrack tid disappeared from current detections.
        # This recovers from ID swaps after brief occlusions.
        for pid, tid in list(self._pid_to_tid.items()):
            if tid not in current_tids:
                del self._tid_to_pid[tid]
                del self._pid_to_tid[pid]
                # Use last-known position as anchor; fall back to initial bbox
                last = self._last_pos.get(pid)
                if last is not None:
                    cx, cy = last
                    bw, bh = 90, 150
                    self._unmatched[pid] = (int(cx - bw/2), int(cy - bh/2), bw, bh)
                else:
                    self._unmatched[pid] = tuple(self._initial_bboxes[pid])

        # Late matching: initial unmatched + re-added lost PIDs
        if self._unmatched:
            self._try_match(dets)

        for i in range(len(dets)):
            tid = dets.tracker_id[i]
            if tid is None:
                continue
            pid = self._tid_to_pid.get(int(tid))
            if pid is None:
                continue
            x1, y1, x2, y2 = dets.xyxy[i]
            dcx, dcy = (x1 + x2) / 2, (y1 + y2) / 2
            # Discard detections outside court + margin (spectators, referees)
            if not self._in_court(dcx, dcy):
                continue
            positions[pid] = (dcx, dcy)
            self._last_pos[pid] = (dcx, dcy)   # update last-known position

        return positions

    @property
    def matched_count(self) -> int:
        return len(self._pid_to_tid)

    @property
    def unmatched_pids(self) -> list:
        return list(self._unmatched.keys())


# ══════════════════════════ EVENT TIMELINE ══════════════════════════════ #

class EventTimeline:
    """
    Stores all match events (winners, errors) with timestamps.
    Used to generate the point-ratio chart in the Excel report.
    """

    def __init__(self):
        # List of dicts: {ts, player_id, player_name, team, kind}
        # kind: 'winner' | 'net_error' | 'out_error'
        self.events: list[dict] = []

    def add(self, ts: float, player_id: int, player_name: str,
            team: str, kind: str):
        self.events.append({
            "ts":     ts,
            "pid":    player_id,
            "name":   player_name,
            "team":   team,
            "kind":   kind,
        })

    def point_ratio_series(self, player_id: int) -> list[tuple]:
        """
        Return a list of (ts, cumulative_ratio) for a player.
        ratio = winners – errors at each event point.
        """
        series = [(0.0, 0.0)]
        cumulative = 0
        for ev in sorted(self.events, key=lambda e: e["ts"]):
            if ev["pid"] != player_id:
                continue
            if ev["kind"] == "winner":
                cumulative += 1
            else:
                cumulative -= 1
            series.append((ev["ts"], float(cumulative)))
        return series


# ════════════════════ EVENT DETECTOR (winners / errors) ═════════════════ #

class EventDetector:
    """
    Detects winners and errors from ball trajectory + player positions.

    Key improvements v3:
    ──────────────────────
    1. GLASS-WALL BOUNCE EXCLUSION (_near_wall):
       Padel is unique – balls can legally bounce off the glass walls.
       When the ball changes direction near the court boundary
       (< WALL_MARGIN px), it's a glass bounce, NOT a player hit →
       last_hitter is NOT updated.

    2. PEAK-SPEED TRACKING ON OPPONENT'S SIDE (_max_speed_opp):
       Instead of a boolean flying-flag, we track the maximum ball speed
       on the opponent's side since crossing the net.
       A winner is recorded only if peak speed > WINNER_MIN_SPEED_OPP.
       This is more robust: a momentary 'slow' frame doesn't lose the
       information that the ball was accelerated.

    3. WINNER CONFIRMATION via N CONSECUTIVE SLOW FRAMES (_slow_frames):
       A winner is not counted after a single 'slow' frame.
       Requires WINNER_SLOW_FRAMES consecutive frames below
       WINNER_BOUNCE_SPEED. Protects against brief speed dips mid-rally.

    4. DIRECTION CHECK FOR NET ERRORS:
       Ball must be moving TOWARD the net (not parallel or away).
       Eliminates false positives from slow balls near the net on the
       opponent's side.

    5. SPEED THRESHOLD FOR OUT ERRORS (OUT_SPEED_MIN):
       Ball must actually be moving (not static detector noise).

    6. INDEPENDENT COOLDOWNS PER EVENT TYPE:
       Net error, out error and winner have separate cooldown timers.
       One error doesn't block a winner from the same rally moment.

    7. EVENT TIMELINE INTEGRATION:
       All detected events are recorded in an EventTimeline instance
       for later use in the point-ratio chart.
    """

    # ── Contact / hitter identification ──────────────────────────────────
    CONTACT_DIST         = 250   # px: ball→player centre to register a hit
    CONTACT_COOLDOWN     = 0.40  # sec: guard against repeated registrations
    WALL_MARGIN          = 25    # px: glass-wall zone (ball touches glass ≈ 0-25px from boundary)
    # Ball must be inside court to register a hit
    # (prevents false positives from detections outside court)
    HIT_INSIDE_COURT     = True

    # ── Net error ─────────────────────────────────────────────────────────
    NET_ZONE_PX          = 70    # px: "near net" zone
    # Speed thresholds are now in px/SECOND (velocity is time-aware)
    NET_STOP_SPEED       = 40.0  # px/s ≈ 0.4 m/s: ball truly stopped near net

    # ── Out error ─────────────────────────────────────────────────────────
    OUT_MARGIN           = -50   # px: must be this far outside court polygon
    OUT_MIN_FRAMES       = 3     # consecutive frames outside court to confirm
    OUT_SPEED_MIN        = 60.0  # px/s ≈ 0.6 m/s: must be moving (not noise)

    # ── Winner ────────────────────────────────────────────────────────────
    WINNER_DIST          = 160   # px: all opponents farther than this → winner
    WINNER_MIN_SPEED_OPP = 250.0 # px/s ≈ 2.5 m/s (9 km/h): minimum shot speed
    WINNER_BOUNCE_SPEED  = 70.0  # px/s ≈ 0.7 m/s: 'ball stopped/bounced'
    WINNER_SLOW_FRAMES   = 2     # consecutive slow frames to confirm winner

    # ── Independent cooldowns ─────────────────────────────────────────────
    COOLDOWN_NET         = 2.0   # sec between net errors
    COOLDOWN_OUT         = 2.0   # sec between out errors
    COOLDOWN_WINNER      = 2.5   # sec between winners

    def __init__(self, calib: CourtCalibration, timeline: EventTimeline,
                 player_names: dict):
        """
        calib        – CourtCalibration instance
        timeline     – EventTimeline to record events
        player_names – {pid: name} mapping for timeline labels
        """
        self.calib        = calib
        self.timeline     = timeline
        self.player_names = player_names
        self.stats: dict  = {}              # pid -> {winners, net_errors, out_errors}
        self._hist: deque = deque(maxlen=30)  # (ts, x, y)

        self._last_hitter:  int | None = None
        self._last_hit_ts:  float      = -999.0
        self._crossed_net:  bool       = False
        self._hitter_side:  int | None = None

        # Independent cooldown timestamps
        self._ts_net:       float = -999.0
        self._ts_out:       float = -999.0
        self._ts_winner:    float = -999.0

        # Winner tracking
        self._max_speed_opp: float = 0.0
        self._slow_frames:   int   = 0

        # Out tracking
        self._out_frames: int = 0

        # ── Diagnostics (never cleared, used for logging) ──────────────────
        self._diag_hitter_set:  int = 0   # times last_hitter was assigned
        self._diag_net_cross:   int = 0   # times ball crossed the net
        self._diag_winner_try:  int = 0   # frames ball was slow on opp side
        self._diag_out_try:     int = 0   # frames ball was outside court

    def register(self, pid: int):
        self.stats.setdefault(pid, {"winners": 0, "net_errors": 0, "out_errors": 0})

    def update(self, ball_px: tuple | None,
               player_px: dict,    # pid -> (px, py) in pixels
               team_map:  dict,    # pid -> team_name
               ts:        float):

        if ball_px:
            self._hist.append((ts, ball_px[0], ball_px[1]))
        else:
            self._out_frames = 0   # ball invisible – reset out counter

        if len(self._hist) < 4:
            return

        if ball_px:
            self._try_set_hitter(ball_px, player_px, ts)
            self._check_crossed_net()
            self._check_net_error(ball_px, ts)
            self._check_out_error(ball_px, ts)
            self._check_winner(ball_px, player_px, team_map, ts)

    # ── Helpers: speed ────────────────────────────────────────────────────

    def _velocity(self, n: int = 3) -> tuple:
        """
        Average velocity vector over the last n history intervals.
        Returns (vx, vy) in px/SECOND (time-aware).
        Using real timestamps makes the result independent of detection rate –
        works correctly whether the ball is detected every frame or every 5th frame.
        """
        pts = list(self._hist)
        if len(pts) < n + 1:
            return (0.0, 0.0)
        vx_list, vy_list = [], []
        for i in range(1, n + 1):
            dt = max(pts[-i][0] - pts[-i-1][0], 1e-6)  # seconds between detections
            vx_list.append((pts[-i][1] - pts[-i-1][1]) / dt)
            vy_list.append((pts[-i][2] - pts[-i-1][2]) / dt)
        return (float(np.mean(vx_list)), float(np.mean(vy_list)))

    def _speed(self, n: int = 3) -> float:
        """Speed in px/second."""
        vx, vy = self._velocity(n)
        return np.hypot(vx, vy)

    # ── Helpers: geometry ─────────────────────────────────────────────────

    def _near_wall(self, bx: float, by: float) -> bool:
        """
        True if the ball is inside the court but within WALL_MARGIN px
        of any boundary.
        A direction change here means a glass bounce, not a player hit.
        pointPolygonTest > 0 inside, value = distance to boundary.
        """
        dist = self.calib.dist_to_boundary(bx, by)
        return 0.0 < dist < self.WALL_MARGIN

    # ── Determine last hitter ──────────────────────────────────────────────

    def _try_set_hitter(self, ball_px, player_px, ts):
        if not player_px:
            return
        if len(self._hist) < 5:
            return

        pts = list(self._hist)

        # Time-aware velocity vectors (px/second).
        # Dividing by dt makes detection-gap-independent direction changes robust.
        dt_old = max(pts[-3][0] - pts[-5][0], 1e-6)
        dt_new = max(pts[-1][0] - pts[-3][0], 1e-6)
        v_old = np.array(
            [(pts[-3][1]-pts[-5][1])/dt_old,
             (pts[-3][2]-pts[-5][2])/dt_old], float)
        v_new = np.array(
            [(pts[-1][1]-pts[-3][1])/dt_new,
             (pts[-1][2]-pts[-3][2])/dt_new], float)
        n_old = np.linalg.norm(v_old)
        n_new = np.linalg.norm(v_new)

        # Minimum 20 px/s in both intervals to detect a real directional change
        if n_old < 20 or n_new < 20:
            return

        cos_a = np.dot(v_old, v_new) / (n_old * n_new)
        if cos_a >= 0.45:      # angle < ~63° → not a hit (slightly more lenient than 60°)
            return

        bx, by = ball_px

        # Only register hits when ball is inside the court
        if self.HIT_INSIDE_COURT and not self.calib.in_court(bx, by):
            return

        # Glass-wall exclusion: direction change near wall → bounce, not hit
        if self._near_wall(bx, by):
            return

        if (ts - self._last_hit_ts) <= self.CONTACT_COOLDOWN:
            return

        nearest, nd = None, float('inf')
        for pid, (px, py) in player_px.items():
            d = np.hypot(bx - px, by - py)
            if d < nd:
                nd, nearest = d, pid

        if nd > self.CONTACT_DIST:
            return

        # Register the hit
        self._diag_hitter_set += 1
        self._last_hitter    = nearest
        self._last_hit_ts    = ts
        self._crossed_net    = False
        self._hitter_side    = self.calib.net_side(bx, by)
        self._out_frames     = 0
        self._max_speed_opp  = 0.0
        self._slow_frames    = 0

    # ── Net crossing ──────────────────────────────────────────────────────

    def _check_crossed_net(self):
        """
        Detect when the ball crosses the net line in the current history window.
        Checks the last 8 history entries (≈ 0.67 s at 12 fps) so that a
        crossing detected a few frames ago is still caught.
        """
        if self._crossed_net:
            return   # already crossed – no need to re-check
        pts = list(self._hist)
        for i in range(max(0, len(pts) - 8), len(pts) - 1):
            p1 = (pts[i][1],   pts[i][2])
            p2 = (pts[i+1][1], pts[i+1][2])
            if self.calib.crosses_net(p1, p2):
                self._diag_net_cross += 1
                self._crossed_net = True
                break

    # ── Net error ─────────────────────────────────────────────────────────

    def _check_net_error(self, ball_px, ts):
        if (ts - self._ts_net) < self.COOLDOWN_NET:
            return
        if self._crossed_net:
            return   # ball crossed the net – this is a rally, not an error

        # Require a known hitter set within 3 seconds
        if self._last_hitter is None:
            return
        if (ts - self._last_hit_ts) > 3.0:
            return

        bx, by = ball_px
        if abs(by - self.calib.net_y(bx)) > self.NET_ZONE_PX:
            return
        if self._speed() >= self.NET_STOP_SPEED:
            return

        # Ball must be on the hitter's side
        if self._hitter_side is not None:
            if self.calib.net_side(bx, by) != self._hitter_side:
                return

        # Ball must be moving TOWARD the net (not parallel/away).
        # Velocity is now in px/second; use 20 px/s (~0.2 m/s) as minimum.
        # net_side=0 → ball above net (lower Y) → toward net means vy > 0 (downward)
        # net_side=1 → ball below net (higher Y) → toward net means vy < 0 (upward)
        _, vy = self._velocity(5)
        if abs(vy) > 20:   # 20 px/s threshold: meaningful directional movement
            if self._hitter_side == 0 and vy < 0:
                return   # moving AWAY from net
            if self._hitter_side == 1 and vy > 0:
                return   # moving AWAY from net

        self._record_error("net_errors", ts, "net")

    # ── Out error ─────────────────────────────────────────────────────────

    def _check_out_error(self, ball_px, ts):
        if (ts - self._ts_out) < self.COOLDOWN_OUT:
            return
        bx, by = ball_px
        dist = self.calib.dist_to_boundary(bx, by)
        if dist < self.OUT_MARGIN:
            if self._speed() >= self.OUT_SPEED_MIN:
                self._out_frames += 1
                self._diag_out_try += 1
                if self._out_frames >= self.OUT_MIN_FRAMES:
                    self._record_error("out_errors", ts, "out")
            else:
                self._out_frames = 0
        else:
            self._out_frames = 0

    # ── Winner ────────────────────────────────────────────────────────────

    def _check_winner(self, ball_px, player_px, team_map, ts):
        if (ts - self._ts_winner) < self.COOLDOWN_WINNER:
            return
        if not self._crossed_net:
            return
        if self._last_hitter is None:
            return

        hitter_team = team_map.get(self._last_hitter)
        if not hitter_team:
            return

        bx, by = ball_px

        # Ball must be on the OPPONENT'S side
        if self._hitter_side is not None:
            if self.calib.net_side(bx, by) == self._hitter_side:
                # Returned to hitter's side → reset winner tracking
                self._max_speed_opp = 0.0
                self._slow_frames   = 0
                return

        spd = self._speed()

        # Accumulate peak speed on opponent's side
        if spd > self._max_speed_opp:
            self._max_speed_opp = spd

        # Count consecutive 'stopped' frames
        if spd <= self.WINNER_BOUNCE_SPEED:
            self._slow_frames += 1
            self._diag_winner_try += 1
        else:
            self._slow_frames = 0
            return   # ball still moving – keep waiting

        # Winner requires:
        # 1. Ball reached minimum peak speed (was a real shot)
        if self._max_speed_opp < self.WINNER_MIN_SPEED_OPP:
            return

        # 2. Ball slow for N consecutive frames (landed / stopped)
        if self._slow_frames < self.WINNER_SLOW_FRAMES:
            return

        # 3. Ball inside court (bounced, didn't fly out)
        if not self.calib.in_court(bx, by):
            return

        # 4. All opponents are far from landing point (skip check if positions unknown)
        opponents = {pid: pos for pid, pos in player_px.items()
                     if team_map.get(pid) != hitter_team}
        if opponents:
            if min(np.hypot(bx-px, by-py) for px, py in opponents.values()) < self.WINNER_DIST:
                return

        # Record winner
        pid  = self._last_hitter
        name = self.player_names.get(pid, str(pid))
        team = hitter_team
        if pid in self.stats:
            self.stats[pid]["winners"] += 1
        self.timeline.add(ts, pid, name, team, "winner")

        self._last_hitter    = None
        self._crossed_net    = False
        self._max_speed_opp  = 0.0
        self._slow_frames    = 0
        self._ts_winner      = ts

    # ── Record error ──────────────────────────────────────────────────────

    def _record_error(self, key: str, ts: float, kind: str):
        pid  = self._last_hitter
        name = self.player_names.get(pid, str(pid)) if pid is not None else "?"
        team = ""  # will be looked up from stats if needed
        if pid is not None and pid in self.stats:
            self.stats[pid][key] += 1
            self.timeline.add(ts, pid, name, team, kind)

        self._last_hitter   = None
        self._crossed_net   = False
        self._max_speed_opp = 0.0
        self._slow_frames   = 0
        if kind == "net":
            self._ts_net = ts
        elif kind == "out":
            self._ts_out = ts
        # Clear history to avoid double-counting the same event
        self._hist.clear()


# ═══════════════════════ SIDE SWITCH DETECTOR ═══════════════════════════ #

class SideSwitchDetector:
    """
    Detects when the two teams swap court sides (after each game set).

    Uses normalised Y-position history.  In a padel video with a nearly
    horizontal net, the "home side" of a player corresponds to whether they
    are consistently in the TOP half (y < 0.5) or BOTTOM half (y > 0.5)
    of the frame.  After a side swap both teams move to the opposite half.

    Using Y (not X) is critical for courts with a horizontal net line.
    """

    STABLE_WINDOW = 8.0    # seconds of history to classify 'home side'
    CONFIRM_SEC   = 1.0    # seconds of consistent inverted positions to confirm swap
    COOLDOWN_SEC  = 90.0   # minimum seconds between consecutive side switches
    DEAD_ZONE     = 0.02   # centre zone excluded from side classification
    SIDE_RATIO    = 0.45   # fraction of clear positions needed to assign a side

    def __init__(self, log_cb=None):
        self.switch_times: list           = []
        self._last_switch                 = -self.COOLDOWN_SEC
        self._history: dict               = {}
        self._home_sides: dict            = {}
        self._candidate_ts                = None
        self._candidate_sides: dict | None = None
        self._log_cb                      = log_cb or (lambda msg: None)

    def update(self, player_data: dict, ts: float) -> bool:
        """
        Call each processed frame with {pid: normalised_y}.
        Returns True when a side switch is confirmed.
        """
        self._update_history(player_data, ts)
        if ts - self._last_switch < self.COOLDOWN_SEC:
            return False
        current = self._compute_sides()

        if len(current) < 2:          # need at least 2 players with clear sides
            return False
        if not self._home_sides:
            self._home_sides = dict(current)
            self._log_cb(f"SideSwitch: home sides established at {ts:.1f}s: {self._home_sides}")
            return False
        return self._check_switch(current, ts)

    def _update_history(self, player_data, ts):
        cutoff = ts - self.STABLE_WINDOW
        for pid, cx in player_data.items():
            self._history.setdefault(pid, []).append((ts, cx))
            self._history[pid] = [(t, x) for t, x in self._history[pid] if t >= cutoff]

    def _compute_sides(self) -> dict:
        """
        Assign each player a side (0 = top half / far end, 1 = bottom half / near end)
        based on their normalised-Y history.
        """
        sides = {}
        for pid, hist in self._history.items():
            if len(hist) < 8:
                continue
            # Discard positions too close to the frame centre (net area)
            clear = [y for _, y in hist if abs(y - 0.5) > self.DEAD_ZONE]
            if len(clear) < 5:
                continue
            top = sum(1 for y in clear if y < 0.5)   # top half (far end)
            if top / len(clear) >= self.SIDE_RATIO:
                sides[pid] = 0   # consistently in top half
            elif (len(clear) - top) / len(clear) >= self.SIDE_RATIO:
                sides[pid] = 1   # consistently in bottom half
        return sides

    def _check_switch(self, current, ts) -> bool:
        common = set(current) & set(self._home_sides)
        if len(common) < 2:
            return False
        all_flipped = all(current[p] != self._home_sides[p] for p in common)
        if not all_flipped:
            self._candidate_ts = self._candidate_sides = None
            return False
        if self._candidate_ts is None:
            self._candidate_ts    = ts
            self._candidate_sides = dict(current)
            return False
        consistent = all(current.get(p) == s
                         for p, s in self._candidate_sides.items() if p in current)
        if not consistent:
            self._candidate_ts = self._candidate_sides = None
            return False
        if ts - self._candidate_ts >= self.CONFIRM_SEC:
            self.switch_times.append(self._candidate_ts)
            self._last_switch     = self._candidate_ts
            self._home_sides      = dict(current)
            self._candidate_ts    = self._candidate_sides = None
            return True
        return False


# ═══════════════════════════ PLAYER TRACKER ═════════════════════════════ #

class PlayerTracker:
    """
    Per-player tracker combining CSRT (fallback) or YOLO+ByteTrack detections
    with movement statistics, zone-time tracking and heatmap generation.
    """

    ZONE_NAMES = {
        "left_back":   "Left Back",
        "center_back": "Centre Back",
        "right_back":  "Right Back",
        "left_mid":    "Left Mid",
        "center_mid":  "Centre Court",
        "right_mid":   "Right Mid",
        "left_net":    "Left Net",
        "center_net":  "Centre Net",
        "right_net":   "Right Net",
    }

    def __init__(self, pid: int, name: str, team: str, bbox, frame):
        self.player_id = pid
        self.name      = name
        self.team      = team
        # CSRT tracker is created as a fallback;
        # when using YOLOv8+ByteTrack it is not used.
        try:
            self.tracker = create_tracker()
            self.tracker.init(frame, bbox)
        except RuntimeError:
            self.tracker = None   # no CSRT available – expect YOLO path

        self.active    = True

        self.timestamps: list = []
        self.zone_time: dict  = {k: 0.0 for k in self.ZONE_NAMES}
        self._heatmap         = np.zeros((5, 5), dtype=int)
        self.distance_px      = 0.0
        self.sprints          = 0
        self.lost_frames      = 0

        # Current positions (updated each processed frame)
        self.last_cx: float | None = None   # normalised (0–1)
        self.last_cy: float | None = None
        self.last_px: float | None = None   # pixels
        self.last_py: float | None = None
        self.last_active_ts: float = -999.0  # timestamp of last real detection

        self._prev_raw: tuple | None = None
        self._prev_ts:  float | None = None
        self._speed_buf: list        = []

        # Per-game (per-period) stats.  Each element = one game.
        self._period_stats: list = [{"distance_px": 0.0, "time": 0.0, "sprints": 0}]

    # ── New period (called on side switch) ───────────────────────────────

    def start_new_period(self):
        """Start a new game period.  Breaks temporal link between periods."""
        self._period_stats.append({"distance_px": 0.0, "time": 0.0, "sprints": 0})
        self._prev_ts  = None   # do not accumulate time across periods
        self._prev_raw = None   # do not accumulate distance across periods

    # ── Shared stats update (used by both CSRT and YOLO paths) ───────────

    def _update_stats(self, raw_cx: float, raw_cy: float, ts: float,
                      fw: int, fh: int, switch_times: list,
                      sprint_thr: float = 70.0):
        self.last_cx = raw_cx
        self.last_cy = raw_cy
        self.last_px = raw_cx * fw
        self.last_active_ts = ts          # record when position was last confirmed
        self.last_py = raw_cy * fh

        # Invert coordinates if court sides are currently swapped
        cx, cy = (1-raw_cx, 1-raw_cy) if is_inverted(ts, switch_times) else (raw_cx, raw_cy)
        self.timestamps.append(ts)

        old_ts = self._prev_ts
        if self._prev_raw and self._prev_ts is not None:
            dx = (raw_cx - self._prev_raw[0]) * fw
            dy = (raw_cy - self._prev_raw[1]) * fh
            dist  = np.hypot(dx, dy)
            dt    = max(ts - self._prev_ts, 1e-6)
            speed = dist / dt
            self._speed_buf.append(speed)
            if len(self._speed_buf) > 5:
                self._speed_buf.pop(0)
            self.distance_px += dist
            self._period_stats[-1]["distance_px"] += dist
            if (speed > sprint_thr and len(self._speed_buf) > 1
                    and self._speed_buf[-2] <= sprint_thr):
                self.sprints += 1
                self._period_stats[-1]["sprints"] += 1

        self._prev_raw = (raw_cx, raw_cy)
        self._prev_ts  = ts

        # Zone classification (3×3 grid: back/mid/net × left/centre/right)
        zone = "back" if cy < 0.33 else "mid" if cy < 0.66 else "net"
        col  = "left" if cx < 0.33 else "center" if cx < 0.66 else "right"
        key  = f"{col}_{zone}"
        if old_ts is not None:
            dt_zone = ts - old_ts
            self.zone_time[key] += dt_zone
            self._period_stats[-1]["time"] += dt_zone

        # 5×5 heatmap
        r = min(int(cy*5), 4)
        c = min(int(cx*5), 4)
        self._heatmap[r, c] += 1
        self.active = True

    # ── CSRT path (used when YOLO is unavailable) ─────────────────────────

    def update(self, frame, ts: float, fw: int, fh: int,
               switch_times: list, sprint_thr: float = 70.0) -> bool:
        if self.tracker is None:
            self.mark_lost()
            return False
        ok, bbox = self.tracker.update(frame)
        if not ok:
            self.lost_frames += 1
            self.active  = False
            self.last_cx = self.last_cy = None
            return False

        raw_cx = (bbox[0] + bbox[2]/2) / fw
        raw_cy = (bbox[1] + bbox[3]/2) / fh
        self._update_stats(raw_cx, raw_cy, ts, fw, fh, switch_times, sprint_thr)
        return True

    # ── YOLO+ByteTrack path ───────────────────────────────────────────────

    def update_from_detection(self, px: float, py: float, ts: float,
                              fw: int, fh: int, switch_times: list,
                              sprint_thr: float = 70.0) -> bool:
        """Update stats from an external detection (pixel centre of bbox)."""
        self._update_stats(px / fw, py / fh, ts, fw, fh, switch_times, sprint_thr)
        return True

    def mark_lost(self):
        """
        Mark player as lost (no detection in this frame).

        The last known pixel position is kept for up to LOST_KEEP_FRAMES
        consecutive lost frames so that event detection (hitter attribution,
        winner proximity check) can still use the last-seen position instead
        of having a gap in player_px.
        """
        self.lost_frames += 1
        self.active = False
        # Keep last-known position for a long window so that hitter attribution
        # and opponent-distance checks still work even during long YOLO gaps.
        # 375 frames ≈ 15 s at 25 fps.  Stale positions are inaccurate but
        # better than no position at all for event detection.
        LOST_KEEP_FRAMES = 375
        if self.lost_frames > LOST_KEEP_FRAMES:
            self.last_cx = self.last_cy = None
            self.last_px = self.last_py = None

    def try_reinit(self, frame, fgmask: np.ndarray,
                   fw: int, fh: int) -> bool:
        """
        Attempt to reinitialise the tracker when the target is lost.
        Finds the nearest human-shaped MOG2 blob within 150 px of the
        last known position and re-initialises the CSRT tracker.
        Returns True on success.
        """
        if self.last_px is None or self.last_py is None:
            return False

        cnts, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        best_blob = None
        best_dist = float("inf")
        for c in cnts:
            if cv2.contourArea(c) < 400:   # too small for a player
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h < w * 0.8:               # horizontal blob → not a person
                continue
            cx, cy = x + w / 2, y + h / 2
            d = np.hypot(cx - self.last_px, cy - self.last_py)
            if d < best_dist and d < 150:
                best_dist = d
                best_blob = (x, y, w, h)

        if best_blob is None:
            return False

        try:
            self.tracker = create_tracker()
            self.tracker.init(frame, best_blob)
        except RuntimeError:
            return False   # no tracker available

        x, y, w, h = best_blob
        self.last_px = x + w / 2
        self.last_py = y + h / 2
        self.last_cx = self.last_px / fw
        self.last_cy = self.last_py / fh
        self.active  = True
        self.lost_frames = max(0, self.lost_frames - 1)
        return True

    def summary(self, event_stats: dict, px_per_meter: float = 10.0) -> dict:
        """Return a summary dict with all statistics for this player."""
        total = (self.timestamps[-1] - self.timestamps[0]) if len(self.timestamps) > 1 else 0.0
        top   = max(self.zone_time, key=self.zone_time.get) if self.zone_time else "-"
        es    = event_stats.get(self.player_id, {})
        errors_total = es.get("net_errors", 0) + es.get("out_errors", 0)
        return {
            "Player":          self.name,
            "Team":            self.team,
            "Winners":         es.get("winners", 0),
            "Errors (total)":  errors_total,
            "Net errors":      es.get("net_errors", 0),
            "Out errors":      es.get("out_errors", 0),
            "Playing time":    str(timedelta(seconds=int(total))),
            "Distance (m)":    round(self.distance_px / max(px_per_meter, 1), 1),
            "Sprints":         self.sprints,
            "Top zone":        self.ZONE_NAMES.get(top, top),
            "Lost (frames)":   self.lost_frames,
            "player_id":       self.player_id,
            "heatmap":         self._heatmap.tolist(),
            "zone_time":       {self.ZONE_NAMES.get(k, k): round(v, 1)
                                for k, v in self.zone_time.items()},
            "per_period":      [
                {"Distance (m)": round(p["distance_px"] / max(px_per_meter, 1), 1),
                 "Time (s)":     round(p["time"], 1),
                 "Sprints":      p["sprints"]}
                for p in self._period_stats
            ],
        }


# ═══════════════════════════ VIDEO ANALYZER ═════════════════════════════ #

class VideoAnalyzer:
    """
    Main analysis loop: reads video frame-by-frame, runs player tracking,
    ball detection, event detection and side-switch detection.
    """

    def __init__(self, video_path: str, players: list,
                 calib: CourtCalibration | None,
                 progress_cb=None, log_cb=None):
        self.video_path  = video_path
        self.players_cfg = players
        self.calib       = calib
        self.progress_cb = progress_cb or (lambda p: None)
        self.log_cb      = log_cb      or (lambda m: None)

    def run(self) -> tuple:
        """
        Run full analysis.
        Returns (summaries, switch_times, timeline) where:
          summaries    – list of per-player stat dicts
          switch_times – list of timestamps (sec) of detected side switches
          timeline     – EventTimeline with all winners/errors
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video file")

        try:
            fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.log_cb(f"Video: {fw}×{fh} @ {fps:.1f}fps  |  {fmt_time(total/fps)}")

            ret, first = cap.read()
            if not ret:
                raise RuntimeError("Could not read first frame")

            # Initialise per-player trackers
            trackers: list[PlayerTracker] = []
            for i, p in enumerate(self.players_cfg):
                t = PlayerTracker(i, p["name"], p["team"], tuple(p["bbox"]), first)
                trackers.append(t)
                self.log_cb(f"Tracker: {p['name']} ({p['team']})")

            # Build helper maps
            player_names = {t.player_id: t.name for t in trackers}
            team_map     = {t.player_id: t.team for t in trackers}

            sw_det   = SideSwitchDetector(log_cb=self.log_cb)
            timeline = EventTimeline()

            court_poly_arr = (np.array(self.calib.corners, dtype=np.float32)
                              if self.calib else None)
            ball_det = BallDetector(court_poly=court_poly_arr) if self.calib else None
            ev_det   = EventDetector(self.calib, timeline, player_names) if self.calib else None
            if ev_det:
                for t in trackers:
                    ev_det.register(t.player_id)
                self.log_cb("Ball detector + Kalman filter active")
            else:
                self.log_cb("WARNING: No calibration – winners/errors will not be counted")

            # ── YOLOv8 + ByteTrack ────────────────────────────────────────
            yolo_tracker = None
            _init_fi     = 1   # frame index where YOLO was initialised (default: 1)
            if _YOLO_AVAILABLE:
                try:
                    self.log_cb("Loading YOLOv8n…")
                    court_poly = (np.array(self.calib.corners, dtype=np.float32)
                                  if self.calib else None)

                    # Load the YOLO model ONCE and reuse across all probe frames
                    import io as _io
                    _se, sys.stderr = sys.stderr, _io.StringIO()
                    try:
                        _yolo_model = _YOLO_CLS("yolov8n.pt")
                    finally:
                        sys.stderr = _se

                    # Probe forward to find a frame where players are on court.
                    # If the first frame has no players (intro/warmup), keep
                    # looking until t=10 s.  The analysis main loop then starts
                    # from the found frame to keep ByteTrack state consistent.
                    _init_fi   = 1    # frame index where main loop will begin
                    yolo_tracker = None
                    for _probe_sec in range(0, 12, 2):
                        _fi_probe = int(_probe_sec * fps)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, _fi_probe)
                        _r, _f = cap.read()
                        if not _r:
                            break
                        _probe_tracker = YOLOByteTracker(
                            [p["bbox"] for p in self.players_cfg], _f,
                            court_poly=court_poly, model=_yolo_model)
                        if _probe_tracker.matched_count > 0:
                            yolo_tracker = _probe_tracker
                            _init_fi     = max(1, _fi_probe)
                            self.log_cb(
                                f"YOLOv8n + ByteTrack: matched "
                                f"{yolo_tracker.matched_count}/{len(trackers)} players "
                                f"(init at t={_probe_sec}s)")
                            break

                    if yolo_tracker is None:
                        # No matches found in any probe frame – create tracker
                        # so late-matching can happen during the main loop.
                        yolo_tracker = YOLOByteTracker(
                            [p["bbox"] for p in self.players_cfg], first,
                            court_poly=court_poly, model=_yolo_model)
                        _init_fi = 1
                        self.log_cb(
                            "YOLOv8n active (no initial match – will match live)")

                    if yolo_tracker.unmatched_pids:
                        names = [trackers[i].name for i in yolo_tracker.unmatched_pids]
                        self.log_cb(f"Searching in later frames: {', '.join(names)}")

                except Exception as exc:
                    self.log_cb(f"YOLO error ({exc}) – using CSRT")
                    yolo_tracker = None
            else:
                self.log_cb("ultralytics/supervision not installed – using CSRT")

            # MOG2 background subtractor for foreground mask.
            # Used by ball detector (removes static background) and
            # by player tracker reinitialisation.
            bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=300, varThreshold=36, detectShadows=False)

            # Pre-train MOG2 on the first few seconds of video so the
            # background model is stable before ball detection starts.
            # This prevents false positives from players' clothing.
            # Use the YOLO init frame as the training endpoint so MOG2 is
            # already trained when the main analysis loop begins.
            _mog2_end = max(int(fps * 3), _init_fi if yolo_tracker else 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _pf in range(_mog2_end):
                _ret2, _frm = cap.read()
                if not _ret2:
                    break
                bg_sub.apply(_frm, learningRate=0.05)

            # Start the main loop from the YOLO init frame.
            # This keeps ByteTrack's state consistent: the tracker was
            # initialised at frame _init_fi and sees sequential frames from
            # that point onward.
            _start_fi = _init_fi if yolo_tracker else 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, _start_fi)

            # Compute px_per_meter from court calibration.
            # Standard padel court width = 10 m.
            if self.calib:
                corners  = self.calib.corners
                top_w    = np.hypot(corners[1][0]-corners[0][0],
                                    corners[1][1]-corners[0][1])
                bot_w    = np.hypot(corners[2][0]-corners[3][0],
                                    corners[2][1]-corners[3][1])
                px_per_meter = float(np.mean([top_w, bot_w])) / 10.0
                self.log_cb(f"Scale: {px_per_meter:.1f} px/m")
            else:
                px_per_meter = 10.0

            step = max(1, int(fps / 12))   # process ~12 fps
            fi   = _start_fi            # start from YOLO init frame
            _ball_det_count  = 0
            _ball_proc_count = 0

            while True:
                ret = cap.grab()
                if not ret:
                    break
                fi += 1
                if fi % step != 0:
                    continue
                ret2, frame = cap.retrieve()
                if not ret2:
                    break

                ts = fi / fps
                sw = sw_det.switch_times

                # Foreground mask (train MOG2 on every processed frame)
                fgmask = bg_sub.apply(frame)
                kern   = np.ones((5, 5), np.uint8)
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN,  kern)
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kern)

                player_px: dict = {}
                player_cx: dict = {}
                player_cy: dict = {}   # normalised Y – used by SideSwitchDetector

                if yolo_tracker is not None:
                    # ── YOLOv8n + ByteTrack path ──────────────────────────
                    positions = yolo_tracker.update(frame)
                    for tr in trackers:
                        if tr.player_id in positions:
                            px, py = positions[tr.player_id]
                            tr.update_from_detection(px, py, ts, fw, fh, sw)
                        else:
                            tr.mark_lost()
                            tr.try_reinit(frame, fgmask, fw, fh)
                else:
                    # ── CSRT fallback path ────────────────────────────────
                    for tr in trackers:
                        ok = tr.update(frame, ts, fw, fh, sw)
                        if not ok:
                            tr.try_reinit(frame, fgmask, fw, fh)

                # player_cy_fresh: only include positions detected within 2 s
                # (stale LOST_KEEP_FRAMES positions confuse the side detector)
                FRESH_WINDOW = 2.0
                player_cy_fresh: dict = {}
                for tr in trackers:
                    if tr.last_cx is not None:
                        player_cx[tr.player_id] = tr.last_cx
                        player_px[tr.player_id] = (tr.last_px, tr.last_py)
                        # Fresh positions for side-switch detection
                        if (ts - tr.last_active_ts) <= FRESH_WINDOW:
                            player_cy[tr.player_id] = tr.last_cy
                            player_cy_fresh[tr.player_id] = tr.last_cy

                # Side switch detection (uses Y position – horizontal-net courts)
                # Only fresh positions used to prevent stale data from masking switches.
                if sw_det.update(player_cy_fresh, ts):
                    self.log_cb(f"Side switch detected: {fmt_time(ts)}")
                    for tr in trackers:
                        tr.start_new_period()

                # Ball detection + event detection
                if ball_det and ev_det:
                    ball = ball_det.detect(frame, fgmask)
                    _ball_proc_count += 1
                    if ball is not None:
                        _ball_det_count += 1
                    ev_det.update(ball, player_px, team_map, ts)

                self.progress_cb(int(fi / total * 100))

        finally:
            cap.release()

        ev_stats  = ev_det.stats if ev_det else {}
        sw_times  = sw_det.switch_times
        summaries = [tr.summary(ev_stats, px_per_meter) for tr in trackers]

        if ball_det and _ball_proc_count > 0:
            det_pct = 100.0 * _ball_det_count / _ball_proc_count
            self.log_cb(
                f"Ball detection rate: {_ball_det_count}/{_ball_proc_count} "
                f"frames ({det_pct:.0f}%)"
            )

        self.log_cb(
            f"Side switches: {len(sw_times)} "
            f"({', '.join(fmt_time(t) for t in sw_times) or 'none'})"
        )
        if ev_det:
            for s in summaries:
                self.log_cb(
                    f"  {s['Player']}: winners={s['Winners']}  "
                    f"errors={s['Errors (total)']} "
                    f"(net={s['Net errors']}, out={s['Out errors']})"
                )
        self.log_cb("Analysis complete")
        return summaries, sw_times, timeline


# ═══════════════════════ REPORT GENERATOR ═══════════════════════════════ #

class ReportGenerator:
    """Generates a multi-sheet Excel report from analysis results."""

    H_FILL   = PatternFill("solid", fgColor="1A3A5C")
    T1_FILL  = PatternFill("solid", fgColor="D6E4F0")
    T2_FILL  = PatternFill("solid", fgColor="FCE4D6")
    WIN_FILL = PatternFill("solid", fgColor="D5F5D5")
    ERR_FILL = PatternFill("solid", fgColor="FFE0E0")
    H_FONT   = Font(bold=True, color="FFFFFF", size=11)
    T_FONT   = Font(bold=True, size=14, color="1A3A5C")
    BORDER   = Border(
        left=Side("thin"), right=Side("thin"),
        top=Side("thin"),  bottom=Side("thin"))

    def generate(self, summaries, video_path, switch_times,
                 timeline: EventTimeline, out_path):
        wb = Workbook()
        self._summary(wb.active, summaries, video_path, switch_times)
        self._by_game(wb.create_sheet("By Game"),           summaries, switch_times)
        self._events( wb.create_sheet("Winners & Errors"),  summaries)
        self._point_ratio(wb.create_sheet("Point Ratio"),   summaries, timeline)
        self._zones(  wb.create_sheet("Zones"),             summaries)
        self._heatmap(wb.create_sheet("Heatmaps"),          summaries)
        wb.save(out_path)

    # ── Summary ───────────────────────────────────────────────────────────

    def _summary(self, ws, summaries, video_path, switch_times):
        ws.title = "Summary"
        ws.merge_cells("A1:I1")
        ws["A1"] = f"Padel Match — {os.path.basename(video_path)}"
        ws["A1"].font      = self.T_FONT
        ws["A1"].alignment = Alignment(horizontal="center")

        sw = ", ".join(fmt_time(t) for t in switch_times) or "none"
        ws.merge_cells("A2:I2")
        ws["A2"] = (f"Generated: {datetime.now().strftime('%d.%m.%Y %H:%M')}   |   "
                    f"Side switches ({len(switch_times)}): {sw}")
        ws["A2"].alignment = Alignment(horizontal="center")

        headers = ["Player", "Team", "Winners", "Errors total",
                   "Net errors", "Out errors",
                   "Playing time", "Distance (m)", "Sprints"]
        for c, h in enumerate(headers, 1):
            cell = ws.cell(row=4, column=c, value=h)
            cell.fill      = self.H_FILL
            cell.font      = self.H_FONT
            cell.alignment = Alignment(horizontal="center")
            cell.border    = self.BORDER

        teams = list(dict.fromkeys(s["Team"] for s in summaries))
        for ri, s in enumerate(summaries, 5):
            base = self.T1_FILL if s["Team"] == teams[0] else self.T2_FILL
            keys = ["Player", "Team", "Winners", "Errors (total)",
                    "Net errors", "Out errors",
                    "Playing time", "Distance (m)", "Sprints"]
            for ci, k in enumerate(keys, 1):
                cell = ws.cell(row=ri, column=ci, value=s[k])
                if   ci == 3:        cell.fill = self.WIN_FILL
                elif ci in (4,5,6):  cell.fill = self.ERR_FILL
                else:                cell.fill = base
                cell.border    = self.BORDER
                cell.alignment = Alignment(horizontal="center")

        for col in ws.columns:
            w = max((len(str(c.value or "")) for c in col), default=10)
            ws.column_dimensions[get_column_letter(col[0].column)].width = w + 3

        # Winners vs Errors bar chart
        n     = len(summaries)
        chart = BarChart()
        chart.type      = "col"
        chart.title     = "Winning points vs Errors"
        chart.style     = 10
        chart.grouping  = "clustered"
        chart.y_axis.title = "Count"
        for col, title in ((3, "Winning points"), (4, "Errors")):
            data = Reference(ws, min_col=col, max_col=col,
                             min_row=4, max_row=4+n)
            chart.add_data(data, titles_from_data=True)
        chart.set_categories(Reference(ws, min_col=1, max_col=1,
                                       min_row=5, max_row=4+n))
        ws.add_chart(chart, f"A{7+n}")

    # ── By Game ───────────────────────────────────────────────────────────

    def _by_game(self, ws, summaries, switch_times):
        """Per-game (per-period) statistics: distance, time, sprints."""
        ws["A1"]      = "Statistics by Game"
        ws["A1"].font = self.T_FONT

        if switch_times:
            periods = []
            prev = "0:00"
            for t in switch_times:
                periods.append(f"Game: {prev} – {fmt_time(t)}")
                prev = fmt_time(t)
            periods.append(f"Game: {prev} – end")
        else:
            periods = ["Game (no side switches detected)"]

        n_periods = max((len(s["per_period"]) for s in summaries), default=1)
        n_periods = max(n_periods, len(periods))

        # Row 2 – game time labels
        for gi, label in enumerate(periods[:n_periods]):
            col_start = 3 + gi * 3
            ws.merge_cells(start_row=2, start_column=col_start,
                           end_row=2,   end_column=col_start + 2)
            cell            = ws.cell(row=2, column=col_start, value=label)
            cell.font       = Font(bold=True, color="FFFFFF")
            cell.fill       = self.H_FILL
            cell.alignment  = Alignment(horizontal="center")

        # Row 3 – column headers
        ws.cell(row=3, column=1, value="Player").font  = Font(bold=True)
        ws.cell(row=3, column=2, value="Team").font    = Font(bold=True)
        sub_headers = ["Distance (m)", "Time", "Sprints"]
        for gi in range(n_periods):
            col_start = 3 + gi * 3
            for ci, h in enumerate(sub_headers):
                cell            = ws.cell(row=3, column=col_start + ci, value=h)
                cell.fill       = self.H_FILL
                cell.font       = self.H_FONT
                cell.alignment  = Alignment(horizontal="center", wrap_text=True)

        # Data rows
        teams = list(dict.fromkeys(s["Team"] for s in summaries))
        for ri, s in enumerate(summaries, 4):
            base = self.T1_FILL if s["Team"] == teams[0] else self.T2_FILL
            ws.cell(row=ri, column=1, value=s["Player"]).fill  = base
            ws.cell(row=ri, column=2, value=s["Team"]).fill    = base
            for gi, pd in enumerate(s["per_period"][:n_periods]):
                col_start = 3 + gi * 3
                t_str = str(timedelta(seconds=int(pd["Time (s)"])))
                for ci, val in enumerate([pd["Distance (m)"], t_str, pd["Sprints"]]):
                    cell            = ws.cell(row=ri, column=col_start + ci, value=val)
                    cell.fill       = base
                    cell.border     = self.BORDER
                    cell.alignment  = Alignment(horizontal="center")

        for col in ws.columns:
            w = max((len(str(c.value or "")) for c in col), default=8)
            ws.column_dimensions[get_column_letter(col[0].column)].width = max(w + 2, 14)

    # ── Winners & Errors ──────────────────────────────────────────────────

    def _events(self, ws, summaries):
        ws["A1"]      = "Winners and Errors Detail"
        ws["A1"].font = self.T_FONT
        headers = ["Player", "Team", "Winners",
                   "Net errors", "Out errors", "Errors total", "W/E ratio"]
        for c, h in enumerate(headers, 1):
            cell            = ws.cell(row=3, column=c, value=h)
            cell.fill       = self.H_FILL
            cell.font       = self.H_FONT
            cell.alignment  = Alignment(horizontal="center", wrap_text=True)
        for ri, s in enumerate(summaries, 4):
            e     = s["Errors (total)"]
            v     = s["Winners"]
            ratio = f"{v/e:.2f}" if e > 0 else "∞" if v > 0 else "—"
            row   = [s["Player"], s["Team"], v,
                     s["Net errors"], s["Out errors"], e, ratio]
            for ci, val in enumerate(row, 1):
                cell            = ws.cell(row=ri, column=ci, value=val)
                cell.alignment  = Alignment(horizontal="center")
                cell.border     = self.BORDER
        for col in ws.columns:
            w = max((len(str(c.value or "")) for c in col), default=10)
            ws.column_dimensions[get_column_letter(col[0].column)].width = w + 3

    # ── Point Ratio ───────────────────────────────────────────────────────

    def _point_ratio(self, ws, summaries, timeline: EventTimeline):
        """
        Point ratio chart: cumulative (winners − errors) per player over time.
        Mirrors the 'Point ratio' chart shown in the reference mobile app.
        """
        ws["A1"]      = "Point Ratio (cumulative winners − errors over time)"
        ws["A1"].font = self.T_FONT

        # Build time-series data for each player
        all_ts   = sorted({ev["ts"] for ev in timeline.events}) or [0.0]
        # Add time 0 and the final timestamp
        if all_ts and all_ts[0] > 0:
            all_ts = [0.0] + all_ts

        # Write time column
        ws.cell(row=3, column=1, value="Time (s)").font = Font(bold=True)
        for ri, t in enumerate(all_ts, 4):
            ws.cell(row=ri, column=1, value=round(t, 1))

        # Write per-player cumulative ratio columns
        for ci, s in enumerate(summaries, 2):
            pid    = s["player_id"]
            pname  = s["Player"]
            series = timeline.point_ratio_series(pid)
            # Interpolate ratio at each all_ts point
            series_dict = dict(series)
            ws.cell(row=3, column=ci, value=pname).font = Font(bold=True)

            cumulative = 0.0
            ratio_vals = []
            for t in all_ts:
                # Find the last event up to this time
                relevant = [s_r for s_r in series if s_r[0] <= t]
                if relevant:
                    cumulative = relevant[-1][1]
                ratio_vals.append(cumulative)
                ws.cell(row=4 + all_ts.index(t), column=ci, value=cumulative)

        n_rows = len(all_ts)

        if n_rows >= 2 and len(summaries) >= 1:
            chart = LineChart()
            chart.title        = "Point Ratio"
            chart.style        = 10
            chart.y_axis.title = "Cumulative score (W−E)"
            chart.x_axis.title = "Time (s)"
            chart.height       = 14
            chart.width        = 28

            for ci, s in enumerate(summaries, 2):
                data = Reference(ws, min_col=ci, max_col=ci,
                                 min_row=3, max_row=3 + n_rows)
                chart.add_data(data, titles_from_data=True)
            chart.set_categories(Reference(ws, min_col=1, max_col=1,
                                           min_row=4, max_row=3 + n_rows))
            ws.add_chart(chart, f"A{6 + n_rows}")

        for col in ws.columns:
            w = max((len(str(c.value or "")) for c in col), default=10)
            ws.column_dimensions[get_column_letter(col[0].column)].width = max(w + 2, 14)

    # ── Zones ─────────────────────────────────────────────────────────────

    def _zones(self, ws, summaries):
        ws["A1"]      = "Time in Zones (seconds)"
        ws["A1"].font = self.T_FONT
        znames = list(next(iter(summaries))["zone_time"].keys())
        ws.cell(row=3, column=1, value="Player").font = Font(bold=True)
        for c, z in enumerate(znames, 2):
            cell            = ws.cell(row=3, column=c, value=z)
            cell.fill       = self.H_FILL
            cell.font       = self.H_FONT
            cell.alignment  = Alignment(horizontal="center", wrap_text=True)
        for ri, s in enumerate(summaries, 4):
            ws.cell(row=ri, column=1, value=s["Player"])
            for c, z in enumerate(znames, 2):
                ws.cell(row=ri, column=c, value=s["zone_time"].get(z, 0))
        for col in ws.columns:
            ws.column_dimensions[get_column_letter(col[0].column)].width = 20

    # ── Heatmaps ──────────────────────────────────────────────────────────

    def _heatmap(self, ws, summaries):
        ws["A1"]      = "Heatmaps (5×5) – home side always at bottom"
        ws["A1"].font = self.T_FONT
        oc = 1
        for s in summaries:
            ws.cell(row=3, column=oc,
                    value=f"{s['Player']} ({s['Team']})").font = Font(bold=True)
            hm = s["heatmap"]
            mx = max(max(r) for r in hm) or 1
            for r, row in enumerate(hm):
                for c, val in enumerate(row):
                    cell            = ws.cell(row=4+r, column=oc+c, value=val)
                    i               = int(255 - (val/mx)*200)
                    cell.fill       = PatternFill("solid", fgColor=f"FF{i:02X}{i:02X}")
                    cell.alignment  = Alignment(horizontal="center")
                    cell.border     = self.BORDER
            ws.cell(row=9, column=oc+2, value="↑ net").font     = Font(italic=True, size=8)
            ws.cell(row=4, column=oc+2, value="↓ back").font    = Font(italic=True, size=8)
            oc += 7


# ═══════════════════════════════ GUI ════════════════════════════════════ #

class App(tk.Tk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.title("Padel Match Analyzer")
        self.resizable(False, False)
        self.configure(bg="#1A3A5C")
        self.video_path = tk.StringVar()
        self.players:  list                   = []
        self.calib:    CourtCalibration | None = None
        self._build_ui()

    def _build_ui(self):
        tk.Label(self, text="🎾  Padel Match Analyzer",
                 bg="#1A3A5C", fg="white",
                 font=("Helvetica", 18, "bold")).pack(pady=(16, 4))
        tk.Label(self,
                 text="Winners · Errors · Zones · Heatmaps · Point Ratio",
                 bg="#1A3A5C", fg="#A0C4E8",
                 font=("Helvetica", 10)).pack(pady=(0, 12))

        card = tk.Frame(self, bg="white")
        card.pack(padx=24, pady=4, fill="x")

        # 1. Video file
        self._section(card, "1. Video file")
        r = tk.Frame(card, bg="white"); r.pack(fill="x", padx=12, pady=4)
        tk.Entry(r, textvariable=self.video_path, width=44,
                 font=("Helvetica", 10)).pack(side="left", padx=(0, 8))
        self._btn(r, "Browse…", self._browse, width=10).pack(side="left")

        # 2. Court calibration
        self._section(card, "2. Court calibration (required for winners/errors)")
        cr = tk.Frame(card, bg="white"); cr.pack(fill="x", padx=12, pady=4)
        self._btn(cr, "📐 Calibrate court", self._calibrate).pack(side="left")
        self.calib_lbl = tk.Label(cr, text="Not set",
                                  bg="white", fg="#E74C3C",
                                  font=("Helvetica", 10))
        self.calib_lbl.pack(side="left", padx=10)

        # 3. Players
        self._section(card, "3. Players (up to 4)")
        pr = tk.Frame(card, bg="white"); pr.pack(fill="x", padx=12, pady=4)
        self._btn(pr, "+ Add player", self._add_player).pack(side="left")
        self._btn(pr, "✕ Remove last", self._del_player,
                  color="#E74C3C").pack(side="left", padx=8)
        self.plist = tk.Listbox(card, height=4, font=("Helvetica", 10))
        self.plist.pack(fill="x", padx=12, pady=(0, 8))

        # 4. Log
        self._section(card, "4. Log")
        self.log = tk.Text(card, height=7, state="disabled",
                           font=("Courier", 9), bg="#F0F4F8")
        self.log.pack(fill="x", padx=12, pady=(0, 8))
        self.bar = ttk.Progressbar(card, length=420, mode="determinate")
        self.bar.pack(padx=12, pady=(0, 12))

        self._start_btn = tk.Button(
            self, text="▶  Start analysis", command=self._start,
            font=("Helvetica", 13, "bold"), bg="#27AE60", fg="white",
            pady=10, padx=30, relief="flat", cursor="hand2",
            activebackground="#229954", activeforeground="white",
            disabledforeground="#cccccc",
        )
        self._start_btn.pack(pady=12)

    def _section(self, p, t):
        tk.Label(p, text=t, bg="white", fg="#1A3A5C",
                 font=("Helvetica", 11, "bold"),
                 anchor="w").pack(fill="x", padx=12, pady=(10, 2))

    def _btn(self, parent, text, cmd, color="#1A3A5C",
             font=("Helvetica", 10), width=None, pady=6, padx=12):
        frame = tk.Frame(parent, bg=color, cursor="hand2")
        lbl   = tk.Label(frame, text=text, bg=color, fg="white",
                         font=font, padx=padx, pady=pady)
        if width:
            lbl.config(width=width)
        lbl.pack()
        def on_enter(_): frame.config(bg="#2C5F8A"); lbl.config(bg="#2C5F8A")
        def on_leave(_): frame.config(bg=color);    lbl.config(bg=color)
        def on_click(_): cmd()
        for w in (frame, lbl):
            w.bind("<Enter>",    on_enter)
            w.bind("<Leave>",    on_leave)
            w.bind("<Button-1>", on_click)
        return frame

    def _browse(self):
        p = filedialog.askopenfilename(
            filetypes=[("MP4", "*.mp4"),
                       ("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if p:
            self.video_path.set(p)

    def _calibrate(self):
        if not self.video_path.get():
            messagebox.showwarning("Error", "Please select a video file first!")
            return
        self._log("Opening court calibration…")
        calib = calibrate_court(self.video_path.get(), self)
        if calib:
            self.calib = calib
            self.calib_lbl.config(text="✓ Set", fg="#27AE60")
            self._log("✓ Court calibration saved")
        else:
            self._log("Calibration cancelled")

    def _add_player(self):
        if not self.video_path.get():
            messagebox.showwarning("Error", "Please select a video file first!")
            return
        if len(self.players) >= 4:
            messagebox.showinfo("Limit reached", "Maximum 4 players.")
            return

        dlg = tk.Toplevel(self)
        dlg.title("Add player")
        dlg.configure(bg="white")
        dlg.grab_set()

        fields = {}
        for label in ("Player name", "Team"):
            tk.Label(dlg, text=label, bg="white",
                     font=("Helvetica", 10)).pack(padx=20, pady=(10, 2))
            e = tk.Entry(dlg, width=30, font=("Helvetica", 10))
            e.pack(padx=20, pady=(0, 4))
            fields[label] = e

        hint = ("Click on the player in the preview window\n"
                "SPACE/→ – next frame  |  ESC – cancel")
        tk.Label(dlg, text=hint, bg="white", fg="#555",
                 font=("Helvetica", 9), justify="center").pack(padx=20, pady=8)

        def pick():
            name = fields["Player name"].get().strip()
            team = fields["Team"].get().strip()
            if not name or not team:
                messagebox.showwarning("Error", "Please fill all fields!", parent=dlg)
                return

            already = [p["bbox"] for p in self.players]
            bbox = pick_player_on_frame(self.video_path.get(), name, already, self)

            if bbox is None or bbox[2] == 0 or bbox[3] == 0:
                messagebox.showwarning("Cancelled", "Player not selected.", parent=dlg)
                return

            self.players.append({"name": name, "team": team, "bbox": list(bbox)})
            self.plist.insert("end", f"  {name}  [{team}]")
            self._log(f"Added: {name} ({team})")
            dlg.destroy()
            # Automatically prompt for the next player (up to 4)
            if len(self.players) < 4:
                self.after(150, self._add_player)

        self._btn(dlg, "Select on video", pick,
                  font=("Helvetica", 11)).pack(pady=12)

    def _del_player(self):
        if self.players:
            p = self.players.pop()
            self.plist.delete("end")
            self._log(f"Removed: {p['name']}")

    def _start(self):
        if not self.video_path.get():
            messagebox.showwarning("Error", "Please select a video file!"); return
        if not self.players:
            messagebox.showwarning("Error", "Please add at least one player!"); return
        out = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel workbook", "*.xlsx")])
        if not out:
            return
        self.bar["value"] = 0
        self._start_btn.config(state="disabled", bg="#95A5A6",
                               text="⏳  Analysis in progress…")
        threading.Thread(target=self._run, args=(out,), daemon=True).start()

    def _run(self, out):
        try:
            an = VideoAnalyzer(
                self.video_path.get(), self.players, self.calib,
                progress_cb=lambda p: self.after(0, lambda: self.bar.config(value=p)),
                log_cb=lambda m: self.after(0, self._log, m),
            )
            summaries, sw, timeline = an.run()
            ReportGenerator().generate(summaries, self.video_path.get(),
                                       sw, timeline, out)
            self.after(0, self._done, out)
        except Exception as e:
            self.after(0, self._reset_start_btn)
            self.after(0, messagebox.showerror, "Error", str(e))

    def _reset_start_btn(self):
        self._start_btn.config(state="normal", bg="#27AE60",
                               text="▶  Start analysis")

    def _done(self, out):
        self.bar["value"] = 100
        self._reset_start_btn()
        self._log(f"Report saved: {out}")
        if messagebox.askyesno("Done!", f"Report saved:\n{out}\n\nOpen it now?"):
            if os.name == "nt":
                os.startfile(out)
            else:
                subprocess.run(["open", out])   # macOS/Linux – safe, no shell injection


    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")


if __name__ == "__main__":
    App().mainloop()
