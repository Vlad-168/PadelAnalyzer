"""
Padel Match Video Analyzer
---------------------------
Требования:
    pip install opencv-contrib-python numpy openpyxl

Запуск:
    python match_analyzer.py
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

# Опциональные зависимости для YOLOv8 + ByteTrack.
# Подавляем безвредное предупреждение NNPACK (C++-уровень, fd=2)
# при первичной инициализации torch.
def _import_yolo_silent():
    """Импортирует ultralytics + supervision с подавлением NNPACK-предупреждения."""
    import os
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd   = os.dup(2)            # сохраняем реальный stderr (fd 2)
    os.dup2(devnull_fd, 2)            # перенаправляем fd 2 → /dev/null
    try:
        from ultralytics import YOLO as _yolo
        import supervision as _sv
        return _yolo, _sv
    finally:
        os.dup2(saved_fd, 2)          # восстанавливаем stderr
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
from openpyxl.chart import BarChart, Reference
from openpyxl.utils import get_column_letter


# ═══════════════════════════ УТИЛИТЫ ════════════════════════════════════ #

def _cv_close():
    """
    Надёжное закрытие всех OpenCV-окон.
    На macOS (Cocoa) сам destroyAllWindows() только помечает окно для удаления;
    визуально оно исчезает только после нескольких итераций cv2.waitKey(1),
    которые прокачивают событийный цикл AppKit.
    """
    cv2.destroyAllWindows()
    for _ in range(30):
        cv2.waitKey(1)


def create_tracker():
    """
    Создаёт наилучший доступный однообъектный трекер.
    OpenCV 4.13+ переместил CSRT/KCF — пробуем всё по очереди.
    """
    for f in (
        lambda: cv2.TrackerMIL_create(),           # OpenCV 4.5+  (без legacy)
        lambda: cv2.legacy.TrackerCSRT_create(),   # OpenCV < 4.13 contrib
        lambda: cv2.TrackerCSRT_create(),
        lambda: cv2.legacy.TrackerKCF_create(),
        lambda: cv2.TrackerKCF_create(),
    ):
        try:
            return f()
        except AttributeError:
            continue
    raise RuntimeError("Трекер не найден.\npip install opencv-contrib-python")


def put_text(img: np.ndarray, text: str, pos: tuple,
             color=(0, 220, 255), size: int = 22) -> np.ndarray:
    """
    Рисует текст с поддержкой кириллицы через Pillow.
    img — BGR numpy array (OpenCV). Возвращает изменённую копию.
    """
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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

    # Конвертируем BGR → RGB для цвета
    r, g, b = color[2], color[1], color[0]
    draw.text(pos, text, font=font, fill=(r, g, b))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def fmt_time(sec: float) -> str:
    return f"{int(sec)//60}:{int(sec)%60:02d}"


def pick_player_on_frame(video_path: str, player_name: str,
                         already_bboxes: list | None = None,
                         parent=None) -> tuple | None:
    """
    Интерактивный выбор игрока — чистый Tkinter + PIL (без cv2.imshow).

    • Рисует YOLO-рамки прямо на PIL-изображении в Canvas.
    • Клик внутри рамки → выбран (bbox из YOLO).
    • Клик вне рамок → прямоугольник 80×160 вокруг курсора.
    • Кнопки «←» / «→» или клавиши A/D/ПРОБЕЛ для навигации по кадрам.
    • ESC или «Отмена» — отмена выбора.
    Возвращает (x, y, w, h) в координатах ОРИГИНАЛЬНОГО видео, или None.
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

    # ── YOLO (загружаем один раз) ──────────────────────────────────────────
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
                res = model(frm, classes=[0], conf=0.3, verbose=False)[0]
                det_cache[fn] = [tuple(map(int, b.xyxy[0].tolist()))
                                 for b in res.boxes]
            else:
                det_cache[fn] = []
        return det_cache[fn]

    # ── Масштаб для отображения ────────────────────────────────────────────
    scr_w = (parent.winfo_screenwidth()  if parent else 1280) - 80
    scr_h = (parent.winfo_screenheight() if parent else 800)  - 120
    scale = min(scr_w / max(w0, 1), scr_h / max(h0, 1), 1.0)
    dw, dh = max(1, int(w0 * scale)), max(1, int(h0 * scale))

    # ── Tkinter-окно ──────────────────────────────────────────────────────
    root  = parent or tk._default_root
    dlg   = tk.Toplevel(root)
    dlg.title(f"Выбор игрока: {player_name}")
    dlg.resizable(False, False)
    dlg.grab_set()

    COLORS_PIL = ["#00FF00", "#00C8FF", "#FF6400", "#6400FF"]
    state      = {"idx": 0, "selected": None}
    _photo     = [None]   # предотвращаем GC
    _dets_disp = [[]]     # детекции в display-координатах

    # ── Виджеты ───────────────────────────────────────────────────────────
    info_var  = tk.StringVar(value=f"Выберите '{player_name}' — кликните на игрока")
    frame_var = tk.StringVar(value="")

    tk.Label(dlg, textvariable=info_var,  font=("Helvetica", 11), pady=4).pack()
    tk.Label(dlg, textvariable=frame_var, font=("Helvetica", 9),  fg="#888").pack()

    canvas = tk.Canvas(dlg, width=dw, height=dh, cursor="crosshair", bg="black")
    canvas.pack()

    nav = tk.Frame(dlg); nav.pack(pady=6)
    tk.Button(nav, text="← Пред.", command=lambda: navigate(-1)).pack(side="left", padx=4)
    tk.Button(nav, text="След. →", command=lambda: navigate(+1)).pack(side="left", padx=4)
    tk.Button(nav, text="✕ Отмена", command=dlg.destroy).pack(side="left", padx=4)

    # ── Отрисовка кадра ───────────────────────────────────────────────────
    def render(fn: int, frame: np.ndarray):
        dets_orig = get_dets(fn, frame)
        disp_dets = [(int(x1*scale), int(y1*scale),
                      int(x2*scale), int(y2*scale))
                     for x1, y1, x2, y2 in dets_orig]
        _dets_disp[0] = disp_dets

        small = cv2.resize(frame, (dw, dh))
        pil   = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        draw  = ImageDraw.Draw(pil)

        # Уже выбранные игроки — серые рамки
        for bx, by, bw, bh in already_bboxes:
            sx1, sy1 = int(bx*scale), int(by*scale)
            sx2, sy2 = int((bx+bw)*scale), int((by+bh)*scale)
            draw.rectangle([sx1, sy1, sx2, sy2], outline="#787878", width=2)
            draw.text((sx1+4, sy1+4), "выбран", fill="#A0A0A0")

        # YOLO-рамки
        for di, (sx1, sy1, sx2, sy2) in enumerate(disp_dets):
            col = COLORS_PIL[di % len(COLORS_PIL)]
            draw.rectangle([sx1, sy1, sx2, sy2], outline=col, width=3)
            draw.text((sx1+6, sy1+6), f"#{di+1}", fill=col)

        n    = len(disp_dets)
        hint = "Кликните на игрока" if n else "Не найдено — смените кадр"
        info_var.set(f"{hint}  —  '{player_name}'  ({n} найдено)")
        frame_var.set(f"Кадр {fn+1} / {total}   (← / → для навигации по кадрам)")

        photo = ImageTk.PhotoImage(pil)
        canvas.create_image(0, 0, anchor="nw", image=photo)
        _photo[0] = photo   # держим ссылку

    # ── Загрузка кадра ────────────────────────────────────────────────────
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

    # ── Клик по canvas ────────────────────────────────────────────────────
    def on_click(event):
        x, y = event.x, event.y
        for sx1, sy1, sx2, sy2 in _dets_disp[0]:
            if sx1 <= x <= sx2 and sy1 <= y <= sy2:
                ox1 = int(sx1 / scale); oy1 = int(sy1 / scale)
                ox2 = int(sx2 / scale); oy2 = int(sy2 / scale)
                state["selected"] = (ox1, oy1, ox2 - ox1, oy2 - oy1)
                dlg.destroy(); return
        # Клик вне рамок — bbox вокруг курсора
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
    return sum(1 for t in switch_times if ts >= t) % 2 == 1


def seg_intersect(p1, p2, p3, p4) -> bool:
    """Пересекаются ли отрезки p1-p2 и p3-p4."""
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    d1, d2 = cross(p3, p4, p1), cross(p3, p4, p2)
    d3, d4 = cross(p1, p2, p3), cross(p1, p2, p4)
    return (
        ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and
        ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))
    )


# ═══════════════════════ КАЛИБРОВКА КОРТА ═══════════════════════════════ #

class CourtCalibration:
    """4 угла корта + линия сетки в пикселях."""

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
        """0 = выше сетки, 1 = ниже сетки (относительно Y-оси видео)."""
        return 0 if y < self.net_y(x) else 1


def calibrate_court(video_path: str, parent=None) -> CourtCalibration | None:
    """
    Калибровка корта — чистый Tkinter + PIL (без cv2.imshow).

    Пользователь кликает 6 точек в Toplevel-окне:
      1-4: углы корта (по часовой, начиная с верхнего-левого)
      5-6: левый и правый концы сетки
    ESC / кнопка «Отмена» = отмена.
    После 6-го клика окно автоматически закрывается через 700 мс.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    h0, w0 = frame.shape[:2]

    # ── Масштаб для отображения ────────────────────────────────────────────
    root_win = parent or tk._default_root
    scr_w = (parent.winfo_screenwidth()  if parent else 1280) - 80
    scr_h = (parent.winfo_screenheight() if parent else 800)  - 140
    scale = min(scr_w / max(w0, 1), scr_h / max(h0, 1), 1.0)
    dw, dh = max(1, int(w0 * scale)), max(1, int(h0 * scale))

    labels = [
        "1/6  Верхний-ЛЕВЫЙ угол корта",
        "2/6  Верхний-ПРАВЫЙ угол корта",
        "3/6  Нижний-ПРАВЫЙ угол корта",
        "4/6  Нижний-ЛЕВЫЙ угол корта",
        "5/6  ЛЕВЫЙ конец сетки",
        "6/6  ПРАВЫЙ конец сетки",
    ]

    pts_orig: list  = []          # точки в координатах оригинального видео
    state           = {"result": None}
    _photo          = [None]      # предотвращаем GC PhotoImage

    # ── Tkinter-окно ──────────────────────────────────────────────────────
    dlg = tk.Toplevel(root_win)
    dlg.title("Калибровка — кликайте точки по порядку  |  ESC = отмена")
    dlg.resizable(False, False)
    dlg.grab_set()

    info_var = tk.StringVar(value=labels[0])
    tk.Label(dlg, textvariable=info_var,
             font=("Helvetica", 11), pady=6).pack()

    canvas = tk.Canvas(dlg, width=dw, height=dh, cursor="crosshair", bg="black")
    canvas.pack()

    tk.Button(dlg, text="✕ Отмена", command=dlg.destroy,
              font=("Helvetica", 10)).pack(pady=6)

    # Базовый PIL-кадр (масштабированный, без разметки)
    frame_small = cv2.resize(frame, (dw, dh))
    base_pil = Image.fromarray(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))

    # ── Отрисовка текущего состояния ──────────────────────────────────────
    def redraw(done: bool = False):
        pil  = base_pil.copy()
        draw = ImageDraw.Draw(pil)

        disp = [(int(ox * scale), int(oy * scale)) for ox, oy in pts_orig]
        n    = len(disp)

        # Линии контура корта
        if n >= 2:
            for i in range(1, min(n, 4)):
                draw.line([disp[i-1], disp[i]], fill="#00FF00", width=2)
        if n >= 4:
            draw.line([disp[3], disp[0]], fill="#00FF00", width=2)
        # Линия сетки
        if n == 6:
            draw.line([disp[4], disp[5]], fill="#FF4040", width=3)

        # Кружки с номерами
        for i, (px, py) in enumerate(disp):
            r   = 7
            col = "#FF4040" if i >= 4 else "#00FF00"
            draw.ellipse([px - r, py - r, px + r, py + r],
                         fill=col, outline=col)
            draw.text((px + 10, py - 12), str(i + 1), fill=col)

        # Инструкция / статус
        if done:
            draw.text((20, 14), "✓ Все точки выбраны — закрываю…",
                      fill="#00FF00")
        else:
            draw.text((20, 14),
                      labels[min(n, 5)],
                      fill="#00DCFF")

        photo = ImageTk.PhotoImage(pil)
        canvas.create_image(0, 0, anchor="nw", image=photo)
        _photo[0] = photo   # держим ссылку

    # ── Обработчик клика ──────────────────────────────────────────────────
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
            info_var.set("✓ Все точки выбраны!")
            state["result"] = CourtCalibration(
                pts_orig[:4], pts_orig[4], pts_orig[5])
            redraw(done=True)
            dlg.after(700, dlg.destroy)

    canvas.bind("<Button-1>", on_click)
    dlg.bind("<Escape>", lambda e: dlg.destroy())

    redraw()
    dlg.wait_window()
    return state["result"]


# ═══════════════════════ ДЕТЕКТОР МЯЧА ══════════════════════════════════ #

class BallDetector:
    """
    Детектирует мяч через HSV-фильтрацию + анализ контуров.

    Улучшения по сравнению с базовым подходом:
    ─────────────────────────────────────────
    1. КАЛМАН-ФИЛЬТР (4D: x, y, vx, vy):
       На каждом кадре предсказывается следующая позиция мяча.
       Кандидаты за пределами «зоны доверия» (GATE_PX) отбрасываются,
       что резко снижает количество ложных срабатываний от рекламных
       щитов, одежды и разметки корта.
       После детектирования фильтр корректируется измерением.

    2. МАСКА ПЕРЕДНЕГО ПЛАНА (MOG2):
       Если VideoAnalyzer передаёт fgmask от BackgroundSubtractorMOG2,
       HSV-маска AND-уется с ней — статичный фон (корт, сетка) полностью
       исключается до поиска контуров.

    3. FALLBACK:
       Если в кадре нет подходящих кандидатов, фильтр продолжает
       предсказывать (без correction), оставаясь «тёплым» до 8 кадров.
       Позиция в этом случае не возвращается, чтобы не засорять историю.
    """

    HSV_RANGES = [
        (np.array([18, 55,  70]),  np.array([50, 255, 255])),  # жёлто-зелёный
        (np.array([15, 40,  60]),  np.array([55, 255, 255])),  # шире (тени)
    ]
    MIN_AREA  = 15     # снижено с 12: меньше шума от мелких артефактов
    MAX_AREA  = 2500
    MIN_CIRC  = 0.40
    GATE_PX   = 120    # снижено с 180: тесный гейт Калмана → меньше ложных срабатываний

    def __init__(self):
        self._prev:      tuple | None = None
        self._kf        = self._init_kf()
        self._kf_active = False
        self._miss_cnt  = 0    # кадры без детектирования подряд

    @staticmethod
    def _init_kf():
        """4D Kalman filter: state=(x,y,vx,vy), measurement=(x,y)."""
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov    = np.eye(4, dtype=np.float32) * 0.03  # плавнее (было 0.05)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        kf.errorCovPost       = np.eye(4, dtype=np.float32) * 500.0
        return kf

    def detect(self, frame, fgmask: np.ndarray | None = None) -> tuple | None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Предсказываем следующую позицию мяча (Kalman predict)
        gate_center = None
        if self._kf_active:
            pred        = self._kf.predict()
            gate_center = (float(pred[0]), float(pred[1]))

        best_ball  = None
        best_score = 0.0

        for lo, hi in self.HSV_RANGES:
            mask = cv2.inRange(hsv, lo, hi)
            # Исключаем статичный фон маской переднего плана
            if fgmask is not None:
                mask = cv2.bitwise_and(mask, fgmask)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    np.ones((3, 3), np.uint8))
            mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
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

                # Гейтинг по предсказанию Калмана
                if gate_center is not None:
                    gate_dist = np.hypot(cx - gate_center[0], cy - gate_center[1])
                    if gate_dist > self.GATE_PX:
                        continue   # вне зоны доверия — игнорируем
                    prox = max(0.0, 1.0 - gate_dist / self.GATE_PX)
                elif self._prev:
                    d    = np.hypot(cx - self._prev[0], cy - self._prev[1])
                    prox = max(0.0, 1.0 - d / 250.0)
                else:
                    prox = 0.0

                score = circ * 0.55 + prox * 0.45
                if score > best_score:
                    best_score = score
                    best_ball  = (cx, cy)

        # Обновляем Калман-фильтр
        if best_ball is not None:
            meas = np.array([[np.float32(best_ball[0])],
                             [np.float32(best_ball[1])]])
            if not self._kf_active:
                # Инициализируем состояние при первом детектировании
                self._kf.statePost = np.array(
                    [[best_ball[0]], [best_ball[1]], [0.0], [0.0]], np.float32)
                self._kf_active = True
            self._kf.correct(meas)
            self._miss_cnt = 0
        elif self._kf_active:
            # Нет детектирования — фильтр уже вызвал predict() выше;
            # сбрасываем фильтр после 8 пропущенных кадров подряд
            self._miss_cnt += 1
            if self._miss_cnt > 8:
                self._kf_active = False
                self._miss_cnt  = 0

        self._prev = best_ball
        return best_ball


# ══════════════════ YOLO + BYTETRACK ТРЕКЕР ИГРОКОВ ═════════════════════ #

class YOLOByteTracker:
    """
    Многообъектный трекер игроков: YOLOv8n (детектор) + ByteTrack (MOT).

    Принцип работы:
    ───────────────
    1. При инициализации YOLO обнаруживает людей на первом кадре.
       Каждый начальный bbox (ROI от пользователя) сопоставляется
       с ближайшей детекцией → устанавливается связь track_id ↔ pid.

    2. На каждом кадре YOLO снова находит всех людей.
       ByteTrack присваивает им стабильные track_id через время.
       Мы возвращаем {pid: (px_pixels, py_pixels)} для каждого игрока.

    Преимущества перед CSRT:
    ─────────────────────────
    • Не накапливает дрейф — каждый кадр детекция свежая
    • ByteTrack переживает перекрытия и быстрые движения
    • Автопереобнаружение: потерянный игрок находится снова
    • ID-свап двух игроков невозможен
    """

    CONF         = 0.35   # минимальная уверенность YOLO
    BUFFER       = 50     # кадров до удаления потерянного трека
    FPS          = 12     # совпадает со step VideoAnalyzer (~12fps)
    MAX_MATCH_PX = 250    # макс. расстояние (px) для первичного сопоставления

    def __init__(self, initial_bboxes: list, first_frame: np.ndarray):
        # Загружаем модель с подавлением предупреждения NNPACK
        # (безвредное C++-сообщение о неподдерживаемом железе)
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
            # supervision < 0.21 не имеет frame_rate
            self._tracker = sv.ByteTrack(lost_track_buffer=self.BUFFER)

        self._tid_to_pid: dict[int, int] = {}
        self._pid_to_tid: dict[int, int] = {}
        # Изначально все игроки считаются «нераспознанными»
        self._unmatched: dict[int, tuple] = {
            pid: bbox for pid, bbox in enumerate(initial_bboxes)}
        self._match_initial(first_frame)

    # ── Детектирование ──────────────────────────────────────────────────

    def _detect(self, frame: np.ndarray) -> "sv.Detections":
        results = self._model(frame, classes=[0], conf=self.CONF,
                              verbose=False)[0]
        dets = sv.Detections.from_ultralytics(results)
        return self._tracker.update_with_detections(dets)

    # ── Попытка сопоставления bbox → track_id ───────────────────────────

    def _try_match(self, dets: "sv.Detections"):
        """Сопоставить нераспознанных игроков с ближайшими детекциями."""
        if not self._unmatched or len(dets) == 0 or dets.tracker_id is None:
            return
        for pid, (bx, by, bw, bh) in list(self._unmatched.items()):
            icx = bx + bw / 2
            icy = by + bh / 2
            best_tid, best_d = None, float("inf")
            for i in range(len(dets)):
                tid = dets.tracker_id[i]
                if tid is None:
                    continue
                tid = int(tid)
                if tid in self._tid_to_pid:
                    continue   # этот track_id уже назначен другому игроку
                x1, y1, x2, y2 = dets.xyxy[i]
                dcx, dcy = (x1 + x2) / 2, (y1 + y2) / 2
                d = np.hypot(dcx - icx, dcy - icy)
                if d < best_d:
                    best_d, best_tid = d, tid

            if best_tid is not None and best_d < self.MAX_MATCH_PX:
                self._tid_to_pid[best_tid] = pid
                self._pid_to_tid[pid]      = best_tid
                del self._unmatched[pid]

    def _match_initial(self, frame: np.ndarray):
        dets = self._detect(frame)
        self._try_match(dets)

    # ── Обновление на каждом кадре ──────────────────────────────────────

    def update(self, frame: np.ndarray) -> dict:
        """
        Возвращает {pid: (px, py)} — центр bbox в пикселях для каждого
        найденного игрока.  Автоматически дораспознаёт игроков, которых
        не удалось найти на первом кадре (позднее сопоставление).
        """
        dets = self._detect(frame)
        positions: dict[int, tuple] = {}

        if len(dets) == 0 or dets.tracker_id is None:
            return positions

        # Позднее сопоставление нераспознанных игроков
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
            positions[pid] = ((x1 + x2) / 2, (y1 + y2) / 2)

        return positions

    @property
    def matched_count(self) -> int:
        return len(self._pid_to_tid)

    @property
    def unmatched_pids(self) -> list:
        return list(self._unmatched.keys())


# ════════════════════ ДЕТЕКТОР СОБЫТИЙ (виннеры/ошибки) ═════════════════ #

class EventDetector:
    """
    Определяет виннеры и ошибки по траектории мяча + позициям игроков.

    Логика:
    ─────
    ПОСЛЕДНИЙ ИГРОК (last_hitter):
        Игрок, который был ближе всех к мячу в момент, когда мяч
        резко изменил направление (удар или отскок от стекла).
        Обновляется не чаще чем раз в 0.4 сек.

    ОШИБКА В СЕТКУ (net_error):
        Вектор последних N позиций мяча пересекает линию сетки И
        мяч резко замедляется (speed < NET_STOP_SPEED) → мяч застрял в сетке.

    ОШИБКА В АУТ (out_error):
        Позиция мяча оказывается за пределами полигона корта И
        мяч при этом не летел через стену (скорость достаточная → это не
        отскок от стекла; нельзя определить точно, но делаем cooldown).

    ВИННЕР:
        Мяч приземлился на стороне соперника (crossed_net=True) И
        скорость упала (отскок/остановка) И все соперники были
        дальше WINNER_DIST пикселей в момент остановки мяча.
    """

    CONTACT_DIST     = 120   # px (было 85): YOLO даёт центр тела, ракетка дальше
    CONTACT_COOLDOWN = 0.6   # сек (было 0.4)
    NET_STOP_SPEED   = 1.5   # px/frame (было 4.0): мяч должен действительно замереть у сетки
    NET_ZONE_PX      = 40    # px (было 55): строже — сетка — конкретная линия
    WINNER_DIST      = 200   # px (было 130): на падел-корте соперники могут быть далеко
    OUT_MARGIN       = -30   # px (было -12): исключаем шум трекера — мяч явно за кортом
    OUT_MIN_FRAMES   = 2     # кадров подряд за кортом → только тогда аут
    SPEED_BOUNCE     = 4.0   # px/frame (было 8.0): мяч замедлился после отскока
    SPEED_FLYING     = 15.0  # px/frame: новый — мяч считается «в полёте» выше этого порога
    EVENT_COOLDOWN   = 2.0   # сек (было 1.5): минимум между событиями

    def __init__(self, calib: CourtCalibration):
        self.calib  = calib
        self.stats: dict = {}   # pid -> {winners, net_errors, out_errors}

        self._hist: deque = deque(maxlen=30)   # (ts, x, y)
        self._last_hitter:  int | None = None
        self._last_hit_ts:  float      = -999.0
        self._crossed_net:  bool       = False  # мяч перелетел на сторону соперника
        self._event_cooldown: float    = -999.0 # ts последнего зафиксированного события
        # ── Новые поля для точного определения событий ──────────────────
        self._hitter_side:  int | None = None   # net_side() бьющего в момент удара
        self._out_frames:   int        = 0      # кадров подряд мяч за кортом
        self._was_flying:   bool       = False  # мяч был в полёте на стороне соперника

    def register(self, pid: int):
        self.stats.setdefault(pid, {"winners": 0, "net_errors": 0, "out_errors": 0})

    def update(self, ball_px: tuple | None,
               player_px: dict,    # pid -> (px, py)  в пикселях
               team_map:  dict,    # pid -> team_name
               ts:        float):

        if ball_px:
            self._hist.append((ts, ball_px[0], ball_px[1]))
        else:
            self._out_frames = 0   # мяч не виден — не накапливаем счётчик аута

        if len(self._hist) < 4:
            return

        if ball_px:
            self._try_set_hitter(ball_px, player_px, ts)
            self._check_crossed_net(ts)
            self._check_net_error(ball_px, ts)
            self._check_out_error(ball_px, ts)
            self._check_winner(ball_px, player_px, team_map, ts)

    # ── Вспомогательные методы ─────────────────────────────────────────

    def _velocity(self, n: int = 3) -> tuple:
        """Средний вектор скорости по последним n точкам."""
        pts = list(self._hist)
        if len(pts) < n + 1:
            return (0.0, 0.0)
        vx = np.mean([pts[-i][1] - pts[-i-1][1] for i in range(1, n+1)])
        vy = np.mean([pts[-i][2] - pts[-i-1][2] for i in range(1, n+1)])
        return (float(vx), float(vy))

    def _speed(self) -> float:
        vx, vy = self._velocity()
        return np.hypot(vx, vy)

    def _event_allowed(self, ts: float, cooldown: float = 2.0) -> bool:
        return (ts - self._event_cooldown) >= cooldown

    # ── Определение last_hitter ────────────────────────────────────────

    def _try_set_hitter(self, ball_px, player_px, ts):
        if not player_px:
            return
        # Определяем, было ли резкое изменение направления
        if len(self._hist) >= 5:
            pts = list(self._hist)
            v_old = np.array([pts[-3][1]-pts[-5][1], pts[-3][2]-pts[-5][2]], float)
            v_new = np.array([pts[-1][1]-pts[-3][1], pts[-1][2]-pts[-3][2]], float)
            n_old = np.linalg.norm(v_old)
            n_new = np.linalg.norm(v_new)
            if n_old > 2 and n_new > 2:
                cos_a = np.dot(v_old, v_new) / (n_old * n_new)
                dir_changed = cos_a < 0.5   # было 0.3 (>72°) → теперь >60°: ловим больше ударов
            else:
                dir_changed = False
        else:
            dir_changed = True

        if not dir_changed:
            return

        bx, by = ball_px
        nearest, nd = None, float('inf')
        for pid, (px, py) in player_px.items():
            d = np.hypot(bx - px, by - py)
            if d < nd:
                nd, nearest = d, pid

        if nd < self.CONTACT_DIST and (ts - self._last_hit_ts) > self.CONTACT_COOLDOWN:
            self._last_hitter  = nearest
            self._last_hit_ts  = ts
            self._crossed_net  = False   # сбрасываем флаг перелёта
            self._was_flying   = False   # новый удар — сбрасываем флаг полёта
            self._hitter_side  = self.calib.net_side(bx, by)   # запоминаем сторону
            self._out_frames   = 0       # сбрасываем счётчик аута

    # ── Перелёт через сетку ────────────────────────────────────────────

    def _check_crossed_net(self, ts):
        pts = list(self._hist)
        for i in range(max(0, len(pts)-4), len(pts)-1):
            p1 = (pts[i][1],   pts[i][2])
            p2 = (pts[i+1][1], pts[i+1][2])
            if self.calib.crosses_net(p1, p2):
                self._crossed_net = True
                break

    # ── Ошибка: сетка ─────────────────────────────────────────────────

    def _check_net_error(self, ball_px, ts):
        if not self._event_allowed(ts):
            return
        # Если мяч уже перелетел через сетку — это ралли, а не ошибка в сетку
        if self._crossed_net:
            return
        bx, by = ball_px
        near_net = abs(by - self.calib.net_y(bx)) < self.NET_ZONE_PX
        if not near_net or self._speed() >= self.NET_STOP_SPEED:
            return
        # Дополнительная проверка: мяч должен быть на стороне бьющего,
        # а не уже перелетел и остановился у сетки со стороны соперника
        if self._hitter_side is not None:
            if self.calib.net_side(bx, by) != self._hitter_side:
                return
        self._record_error("net_errors", ts)

    # ── Ошибка: аут ───────────────────────────────────────────────────

    def _check_out_error(self, ball_px, ts):
        bx, by = ball_px
        dist = cv2.pointPolygonTest(
            np.array(self.calib.corners, dtype=np.float32),
            (float(bx), float(by)), True)
        if dist < self.OUT_MARGIN:
            self._out_frames += 1
            # Аут подтверждаем только при нескольких кадрах подряд за кортом
            # (защита от шума трекера)
            if self._out_frames >= self.OUT_MIN_FRAMES and self._event_allowed(ts):
                self._record_error("out_errors", ts)
        else:
            self._out_frames = 0   # мяч вернулся в корт — сбрасываем счётчик

    # ── Виннер ────────────────────────────────────────────────────────

    def _check_winner(self, ball_px, player_px, team_map, ts):
        if not self._event_allowed(ts):
            return
        if not self._crossed_net or self._last_hitter is None:
            return

        hitter_team = team_map.get(self._last_hitter)
        if not hitter_team:
            return

        bx, by = ball_px

        # Мяч должен быть на стороне СОПЕРНИКА (не бьющего)
        if self._hitter_side is not None:
            if self.calib.net_side(bx, by) == self._hitter_side:
                return

        spd = self._speed()

        # Обновляем флаг «мяч был в полёте» на стороне соперника
        if spd > self.SPEED_FLYING:
            self._was_flying = True

        # Виннер = мяч летел → замедлился после отскока.
        # Если ещё в полёте или ни разу не разогнался — ждём.
        if not self._was_flying or spd > self.SPEED_BOUNCE:
            return

        if not self.calib.in_court(bx, by):
            return

        # Все соперники должны быть далеко от точки посадки
        opponents = {pid: pos for pid, pos in player_px.items()
                     if team_map.get(pid) != hitter_team}
        if not opponents:
            return

        min_dist = min(np.hypot(bx-px, by-py) for px, py in opponents.values())

        if min_dist > self.WINNER_DIST:
            pid = self._last_hitter
            if pid in self.stats:
                self.stats[pid]["winners"] += 1
            self._last_hitter  = None
            self._crossed_net  = False
            self._was_flying   = False
            self._event_cooldown = ts

    # ── Запись ошибки ──────────────────────────────────────────────────

    def _record_error(self, key: str, ts: float):
        if self._last_hitter is not None and self._last_hitter in self.stats:
            self.stats[self._last_hitter][key] += 1
        self._last_hitter    = None
        self._crossed_net    = False
        self._event_cooldown = ts
        # Очищаем историю мяча чтобы не словить дубль
        self._hist.clear()


# ═══════════════════════ ДЕТЕКТОР СМЕНЫ СТОРОН ══════════════════════════ #

class SideSwitchDetector:
    STABLE_WINDOW = 15.0   # уменьшено с 25: меньше истории для построения модели
    CONFIRM_SEC   = 2.0
    COOLDOWN_SEC  = 120.0
    DEAD_ZONE     = 0.05   # уменьшено с 0.08: больше точек считаются «чёткими»
    SIDE_RATIO    = 0.60   # снижено с 0.72: падел — динамичная игра, игроки часто выходят в центр

    def __init__(self):
        self.switch_times: list          = []
        self._last_switch                = -self.COOLDOWN_SEC
        self._history: dict              = {}
        self._home_sides: dict           = {}
        self._candidate_ts               = None
        self._candidate_sides: dict | None = None

    def update(self, player_data: dict, ts: float) -> bool:
        self._update_history(player_data, ts)
        if ts - self._last_switch < self.COOLDOWN_SEC:
            return False
        current = self._compute_sides()
        if len(current) < max(2, len(player_data) - 1):
            return False
        if not self._home_sides:
            self._home_sides = dict(current)
            return False
        return self._check_switch(current, ts)

    def _update_history(self, player_data, ts):
        cutoff = ts - self.STABLE_WINDOW
        for pid, cx in player_data.items():
            self._history.setdefault(pid, []).append((ts, cx))
            self._history[pid] = [(t, x) for t, x in self._history[pid] if t >= cutoff]

    def _compute_sides(self) -> dict:
        sides = {}
        for pid, hist in self._history.items():
            if len(hist) < 8:       # снижено с 15
                continue
            clear = [x for _, x in hist if abs(x - 0.5) > self.DEAD_ZONE]
            if len(clear) < 5:      # снижено с 10
                continue
            left = sum(1 for x in clear if x < 0.5)
            if left  / len(clear) >= self.SIDE_RATIO: sides[pid] = 0
            elif (len(clear)-left) / len(clear) >= self.SIDE_RATIO: sides[pid] = 1
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


# ═══════════════════════ ТРЕКЕР ИГРОКА ══════════════════════════════════ #

class PlayerTracker:
    ZONE_NAMES = {
        "left_back":   "Левая задняя",
        "center_back": "Центр задней",
        "right_back":  "Правая задняя",
        "left_mid":    "Левая средняя",
        "center_mid":  "Центр корта",
        "right_mid":   "Правая средняя",
        "left_net":    "Левая у сетки",
        "center_net":  "Центр у сетки",
        "right_net":   "Правая у сетки",
    }

    def __init__(self, pid: int, name: str, team: str, bbox, frame):
        self.player_id = pid
        self.name      = name
        self.team      = team
        # CSRT-трекер создаётся как запасной вариант;
        # при работе через YOLOv8+ByteTrack он не используется.
        try:
            self.tracker = create_tracker()
            self.tracker.init(frame, bbox)
        except RuntimeError:
            self.tracker = None   # нет CSRT — ожидаем YOLO
        self.active    = True

        self.timestamps: list  = []
        self.zone_time: dict   = {k: 0.0 for k in self.ZONE_NAMES}
        self._heatmap          = np.zeros((5, 5), dtype=int)
        self.distance_px       = 0.0
        self.sprints           = 0
        self.lost_frames       = 0

        # Текущие позиции (обновляются в update)
        self.last_cx: float | None = None   # нормализованная (0-1), сырая
        self.last_cy: float | None = None
        self.last_px: float | None = None   # пиксельная
        self.last_py: float | None = None

        self._prev_raw: tuple | None = None
        self._prev_ts:  float | None = None
        self._speed_buf: list        = []

        # Статистика по играм (периодам).  Каждый элемент = одна игра.
        self._period_stats: list = [{"distance_px": 0.0, "time": 0.0, "sprints": 0}]

    # ── Смена периода (вызывается при обнаружении смены сторон) ─────────

    def start_new_period(self):
        """Начать новую игру/период.  Обрезает временну́ю связь между периодами."""
        self._period_stats.append({"distance_px": 0.0, "time": 0.0, "sprints": 0})
        self._prev_ts  = None   # не накапливаем время поперёк периодов
        self._prev_raw = None   # не накапливаем дистанцию поперёк периодов

    # ── Общая статистика (вызывается и CSRT, и YOLO путями) ────────────

    def _update_stats(self, raw_cx: float, raw_cy: float, ts: float,
                      fw: int, fh: int, switch_times: list,
                      sprint_thr: float = 70.0):
        self.last_cx = raw_cx
        self.last_cy = raw_cy
        self.last_px = raw_cx * fw
        self.last_py = raw_cy * fh

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
            if len(self._speed_buf) > 5: self._speed_buf.pop(0)
            self.distance_px += dist
            self._period_stats[-1]["distance_px"] += dist
            if (speed > sprint_thr and len(self._speed_buf) > 1
                    and self._speed_buf[-2] <= sprint_thr):
                self.sprints += 1
                self._period_stats[-1]["sprints"] += 1

        self._prev_raw = (raw_cx, raw_cy)
        self._prev_ts  = ts

        zone = ("back" if cy < 0.33 else "mid" if cy < 0.66 else "net")
        col  = ("left" if cx < 0.33 else "center" if cx < 0.66 else "right")
        key  = f"{col}_{zone}"
        if old_ts is not None:
            dt_zone = ts - old_ts
            self.zone_time[key] += dt_zone
            self._period_stats[-1]["time"] += dt_zone

        r = min(int(cy*5), 4)
        c = min(int(cx*5), 4)
        self._heatmap[r, c] += 1
        self.active = True

    # ── CSRT-путь (используется при недоступном YOLO) ───────────────────

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

    # ── YOLOv8+ByteTrack путь ───────────────────────────────────────────

    def update_from_detection(self, px: float, py: float, ts: float,
                              fw: int, fh: int, switch_times: list,
                              sprint_thr: float = 70.0) -> bool:
        """Обновить статистику из внешней детекции (пиксели центра bbox)."""
        self._update_stats(px / fw, py / fh, ts, fw, fh, switch_times, sprint_thr)
        return True

    def mark_lost(self):
        """Пометить игрока как потерянного (нет детекции в кадре)."""
        self.lost_frames += 1
        self.active  = False
        self.last_cx = self.last_cy = None

    def try_reinit(self, frame, fgmask: np.ndarray,
                   fw: int, fh: int) -> bool:
        """
        Пытается переинициализировать трекер, когда он потерял цель.
        Ищет ближайший к последней известной позиции «человекообразный» блоб
        в маске переднего плана (MOG2).  Возвращает True при успехе.
        """
        if self.last_px is None or self.last_py is None:
            return False

        cnts, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        best_blob = None
        best_dist = float("inf")
        for c in cnts:
            if cv2.contourArea(c) < 400:   # слишком мал для игрока
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h < w * 0.8:               # горизонтальный блоб → не человек
                continue
            cx, cy = x + w / 2, y + h / 2
            d = np.hypot(cx - self.last_px, cy - self.last_py)
            if d < best_dist and d < 150:  # не дальше 150px от последней позиции
                best_dist = d
                best_blob = (x, y, w, h)

        if best_blob is None:
            return False

        try:
            self.tracker = create_tracker()
            self.tracker.init(frame, best_blob)
        except RuntimeError:
            return False   # нет CSRT — reinit невозможен
        x, y, w, h = best_blob
        self.last_px = x + w / 2
        self.last_py = y + h / 2
        self.last_cx = self.last_px / fw
        self.last_cy = self.last_py / fh
        self.active  = True
        self.lost_frames = max(0, self.lost_frames - 1)  # не считаем этот кадр потерей
        return True

    def summary(self, event_stats: dict, px_per_meter: float = 10.0) -> dict:
        total = (self.timestamps[-1] - self.timestamps[0]) if len(self.timestamps) > 1 else 0.0
        top   = max(self.zone_time, key=self.zone_time.get) if self.zone_time else "-"
        es    = event_stats.get(self.player_id, {})
        errors_total = es.get("net_errors", 0) + es.get("out_errors", 0)
        return {
            "Игрок":           self.name,
            "Команда":         self.team,
            "Виннеры":         es.get("winners", 0),
            "Ошибки (всего)":  errors_total,
            "Ошибки в сетку":  es.get("net_errors", 0),
            "Ошибки в аут":    es.get("out_errors", 0),
            "Время в игре":    str(timedelta(seconds=int(total))),
            "Дистанция (м)":   round(self.distance_px / max(px_per_meter, 1), 1),
            "Спринтов":        self.sprints,
            "Топ зона":        self.ZONE_NAMES.get(top, top),
            "Потерян (кадр)":  self.lost_frames,
            "heatmap":         self._heatmap.tolist(),
            "zone_time":       {self.ZONE_NAMES.get(k,k): round(v,1)
                                for k,v in self.zone_time.items()},
            "per_period":      [
                {"Дистанция (м)": round(p["distance_px"] / max(px_per_meter, 1), 1),
                 "Время (с)":     round(p["time"], 1),
                 "Спринтов":      p["sprints"]}
                for p in self._period_stats
            ],
        }


# ═══════════════════════ АНАЛИЗАТОР ВИДЕО ═══════════════════════════════ #

class VideoAnalyzer:
    def __init__(self, video_path: str, players: list,
                 calib: CourtCalibration | None,
                 progress_cb=None, log_cb=None):
        self.video_path = video_path
        self.players_cfg = players
        self.calib       = calib
        self.progress_cb = progress_cb or (lambda p: None)
        self.log_cb      = log_cb      or (lambda m: None)

    def run(self) -> tuple:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError("Не удалось открыть видеофайл")

        try:
            fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.log_cb(f"Видео: {fw}×{fh} @ {fps:.1f}fps  |  {fmt_time(total/fps)}")

            ret, first = cap.read()
            if not ret:
                raise RuntimeError("Не удалось прочитать первый кадр")

            trackers: list[PlayerTracker] = []
            for i, p in enumerate(self.players_cfg):
                t = PlayerTracker(i, p["name"], p["team"], tuple(p["bbox"]), first)
                trackers.append(t)
                self.log_cb(f"Трекер: {p['name']} ({p['team']})")

            sw_det   = SideSwitchDetector()
            ball_det = BallDetector() if self.calib else None
            ev_det   = EventDetector(self.calib) if self.calib else None
            if ev_det:
                for t in trackers:
                    ev_det.register(t.player_id)
                self.log_cb("🎾 Детектор мяча + Калман-фильтр активированы")
            else:
                self.log_cb("⚠️  Калибровка не задана — виннеры/ошибки не считаются")

            # ── YOLOv8 + ByteTrack ───────────────────────────────────────
            yolo_tracker = None
            if _YOLO_AVAILABLE:
                try:
                    self.log_cb("⏳ Загружаю YOLOv8n…")
                    yolo_tracker = YOLOByteTracker(
                        [p["bbox"] for p in self.players_cfg], first)
                    n = yolo_tracker.matched_count
                    self.log_cb(
                        f"🤖 YOLOv8n + ByteTrack: совпало {n}/{len(trackers)} игроков")
                    if n == 0:
                        self.log_cb("⚠️  Нет совпадений — откат на CSRT")
                        yolo_tracker = None
                    elif yolo_tracker.unmatched_pids:
                        names = [trackers[i].name
                                 for i in yolo_tracker.unmatched_pids]
                        self.log_cb(
                            f"🔍 Ищу в следующих кадрах: {', '.join(names)}")
                except Exception as exc:
                    self.log_cb(f"⚠️  YOLO ошибка ({exc}) — использую CSRT")
                    yolo_tracker = None
            else:
                self.log_cb("ℹ️  ultralytics/supervision не установлены — использую CSRT")

            # MOG2 background subtractor для маски переднего плана.
            # Используется и детектором мяча (убирает статичный фон),
            # и реинициализацией трекеров игроков.
            bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=40, detectShadows=False)

            # Вычисляем px_per_meter из калибровки корта.
            # Стандартная ширина падел-корта = 10 м.
            if self.calib:
                corners = self.calib.corners
                top_w = np.hypot(corners[1][0] - corners[0][0],
                                 corners[1][1] - corners[0][1])
                bot_w = np.hypot(corners[2][0] - corners[3][0],
                                 corners[2][1] - corners[3][1])
                px_per_meter = float(np.mean([top_w, bot_w])) / 10.0
                self.log_cb(f"Масштаб: {px_per_meter:.1f} px/м")
            else:
                px_per_meter = 10.0

            team_map = {t.player_id: t.team for t in trackers}
            step     = max(1, int(fps / 12))   # ~12fps
            fi       = 1

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

                # Маска переднего плана (обучаем MOG2 на каждом обрабатываемом кадре)
                fgmask = bg_sub.apply(frame)
                # Убираем шум: открытие + замыкание
                kern = np.ones((5, 5), np.uint8)
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kern)
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kern)

                player_px = {}
                player_cx = {}

                if yolo_tracker is not None:
                    # ── YOLOv8n + ByteTrack путь ─────────────────────────
                    positions = yolo_tracker.update(frame)
                    for tr in trackers:
                        if tr.player_id in positions:
                            px, py = positions[tr.player_id]
                            tr.update_from_detection(px, py, ts, fw, fh, sw)
                        else:
                            tr.mark_lost()
                            tr.try_reinit(frame, fgmask, fw, fh)
                else:
                    # ── CSRT fallback путь ───────────────────────────────
                    for tr in trackers:
                        ok = tr.update(frame, ts, fw, fh, sw)
                        if not ok:
                            tr.try_reinit(frame, fgmask, fw, fh)

                for tr in trackers:
                    if tr.last_cx is not None:
                        player_cx[tr.player_id] = tr.last_cx
                        player_px[tr.player_id] = (tr.last_px, tr.last_py)

                # Смена сторон
                if sw_det.update(player_cx, ts):
                    self.log_cb(f"🔄 Смена сторон: {fmt_time(ts)}")
                    for tr in trackers:
                        tr.start_new_period()   # начинаем новый игровой период

                # Мяч + события (передаём fgmask в детектор мяча)
                if ball_det and ev_det:
                    ball = ball_det.detect(frame, fgmask)
                    ev_det.update(ball, player_px, team_map, ts)

                self.progress_cb(int(fi / total * 100))

        finally:
            cap.release()

        ev_stats = ev_det.stats if ev_det else {}
        sw_times = sw_det.switch_times
        summaries = [tr.summary(ev_stats, px_per_meter) for tr in trackers]

        self.log_cb(
            f"Смен сторон: {len(sw_times)} "
            f"({', '.join(fmt_time(t) for t in sw_times) or 'нет'})"
        )
        if ev_det:
            for s in summaries:
                self.log_cb(
                    f"  {s['Игрок']}: виннеры={s['Виннеры']}  "
                    f"ошибки={s['Ошибки (всего)']} "
                    f"(сетка={s['Ошибки в сетку']}, аут={s['Ошибки в аут']})"
                )
        self.log_cb("✅ Анализ завершён")
        return summaries, sw_times


# ═══════════════════════ ГЕНЕРАТОР ОТЧЁТА ═══════════════════════════════ #

class ReportGenerator:
    H_FILL   = PatternFill("solid", fgColor="1A3A5C")
    T1_FILL  = PatternFill("solid", fgColor="D6E4F0")
    T2_FILL  = PatternFill("solid", fgColor="FCE4D6")
    WIN_FILL = PatternFill("solid", fgColor="D5F5D5")
    ERR_FILL = PatternFill("solid", fgColor="FFE0E0")
    H_FONT   = Font(bold=True, color="FFFFFF", size=11)
    T_FONT   = Font(bold=True, size=14, color="1A3A5C")
    BORDER   = Border(*[Side(style="thin")]*0,
                      left=Side("thin"), right=Side("thin"),
                      top=Side("thin"),  bottom=Side("thin"))

    def generate(self, summaries, video_path, switch_times, out_path):
        wb = Workbook()
        self._summary(wb.active,          summaries, video_path, switch_times)
        self._by_game(wb.create_sheet("По играм"),          summaries, switch_times)
        self._events( wb.create_sheet("Виннеры и ошибки"), summaries)
        self._zones(  wb.create_sheet("Зоны"),              summaries)
        self._heatmap(wb.create_sheet("Тепловые карты"),    summaries)
        wb.save(out_path)

    # ── Сводка ────────────────────────────────────────────────────────

    def _summary(self, ws, summaries, video_path, switch_times):
        ws.title = "Сводка"
        ws.merge_cells("A1:I1")
        ws["A1"] = f"Падел-матч — {os.path.basename(video_path)}"
        ws["A1"].font = self.T_FONT
        ws["A1"].alignment = Alignment(horizontal="center")

        sw = ", ".join(fmt_time(t) for t in switch_times) or "нет"
        ws.merge_cells("A2:I2")
        ws["A2"] = (f"Сформировано: {datetime.now().strftime('%d.%m.%Y %H:%M')}   |   "
                    f"Смены сторон ({len(switch_times)}): {sw}")
        ws["A2"].alignment = Alignment(horizontal="center")

        headers = ["Игрок", "Команда", "Виннеры", "Ошибки всего",
                   "Ошибки в сетку", "Ошибки в аут",
                   "Время", "Дистанция (м)", "Спринтов"]
        for c, h in enumerate(headers, 1):
            cell = ws.cell(row=4, column=c, value=h)
            cell.fill = self.H_FILL; cell.font = self.H_FONT
            cell.alignment = Alignment(horizontal="center"); cell.border = self.BORDER

        teams = list(dict.fromkeys(s["Команда"] for s in summaries))
        for ri, s in enumerate(summaries, 5):
            base = self.T1_FILL if s["Команда"] == teams[0] else self.T2_FILL
            keys = ["Игрок","Команда","Виннеры","Ошибки (всего)",
                    "Ошибки в сетку","Ошибки в аут",
                    "Время в игре","Дистанция (м)","Спринтов"]
            for ci, k in enumerate(keys, 1):
                cell = ws.cell(row=ri, column=ci, value=s[k])
                if   ci == 3: cell.fill = self.WIN_FILL
                elif ci in (4,5,6): cell.fill = self.ERR_FILL
                else: cell.fill = base
                cell.border = self.BORDER
                cell.alignment = Alignment(horizontal="center")

        for col in ws.columns:
            w = max((len(str(c.value or "")) for c in col), default=10)
            ws.column_dimensions[get_column_letter(col[0].column)].width = w + 3

        # Диаграмма виннеры vs ошибки
        n = len(summaries)
        chart = BarChart()
        chart.type  = "col"; chart.title = "Виннеры vs Ошибки"
        chart.style = 10;    chart.grouping = "clustered"
        for col, title in ((3, "Виннеры"), (4, "Ошибки")):
            data = Reference(ws, min_col=col, max_col=col, min_row=4, max_row=4+n)
            chart.add_data(data, titles_from_data=True)
        chart.set_categories(Reference(ws, min_col=1, max_col=1, min_row=5, max_row=4+n))
        ws.add_chart(chart, f"A{7+n}")

    # ── По играм ──────────────────────────────────────────────────────

    def _by_game(self, ws, summaries, switch_times):
        """Статистика по отдельным играм (периодам) — дистанция, время, спринты."""
        ws["A1"] = "Статистика по играм"
        ws["A1"].font = self.T_FONT

        # Строка с временны́ми рамками игр
        if switch_times:
            periods = []
            prev = "0:00"
            for t in switch_times:
                periods.append(f"Игра: {prev} – {fmt_time(t)}")
                prev = fmt_time(t)
            periods.append(f"Игра: {prev} – конец")
        else:
            periods = ["Игра (смен не обнаружено)"]

        n_periods = max((len(s["per_period"]) for s in summaries), default=1)
        n_periods = max(n_periods, len(periods))  # на случай расхождения

        # Строка 2 — тайминги игр
        for gi, label in enumerate(periods[:n_periods]):
            col_start = 3 + gi * 3
            ws.merge_cells(start_row=2, start_column=col_start,
                           end_row=2,   end_column=col_start + 2)
            cell = ws.cell(row=2, column=col_start, value=label)
            cell.font      = Font(bold=True, color="FFFFFF")
            cell.fill      = self.H_FILL
            cell.alignment = Alignment(horizontal="center")

        # Строка 3 — заголовки
        ws.cell(row=3, column=1, value="Игрок").font = Font(bold=True)
        ws.cell(row=3, column=2, value="Команда").font = Font(bold=True)
        sub_headers = ["Дистанция (м)", "Время", "Спринтов"]
        for gi in range(n_periods):
            col_start = 3 + gi * 3
            for ci, h in enumerate(sub_headers):
                cell = ws.cell(row=3, column=col_start + ci, value=h)
                cell.fill      = self.H_FILL
                cell.font      = self.H_FONT
                cell.alignment = Alignment(horizontal="center", wrap_text=True)

        # Строки с данными
        teams = list(dict.fromkeys(s["Команда"] for s in summaries))
        for ri, s in enumerate(summaries, 4):
            base = self.T1_FILL if s["Команда"] == teams[0] else self.T2_FILL
            ws.cell(row=ri, column=1, value=s["Игрок"]).fill  = base
            ws.cell(row=ri, column=2, value=s["Команда"]).fill = base
            for gi, pd in enumerate(s["per_period"][:n_periods]):
                col_start = 3 + gi * 3
                t_str = str(timedelta(seconds=int(pd["Время (с)"])))
                for ci, val in enumerate([pd["Дистанция (м)"], t_str, pd["Спринтов"]]):
                    cell = ws.cell(row=ri, column=col_start + ci, value=val)
                    cell.fill      = base
                    cell.border    = self.BORDER
                    cell.alignment = Alignment(horizontal="center")

        # Авто-ширина столбцов
        for col in ws.columns:
            w = max((len(str(c.value or "")) for c in col), default=8)
            ws.column_dimensions[get_column_letter(col[0].column)].width = max(w + 2, 14)

    # ── Виннеры и ошибки ──────────────────────────────────────────────

    def _events(self, ws, summaries):
        ws["A1"] = "Детализация виннеров и ошибок"
        ws["A1"].font = self.T_FONT
        headers = ["Игрок", "Команда", "Виннеры",
                   "Ошибки в сетку", "Ошибки в аут", "Ошибки всего", "В/О соотношение"]
        for c, h in enumerate(headers, 1):
            cell = ws.cell(row=3, column=c, value=h)
            cell.fill = self.H_FILL; cell.font = self.H_FONT
            cell.alignment = Alignment(horizontal="center", wrap_text=True)
        for ri, s in enumerate(summaries, 4):
            e = s["Ошибки (всего)"]
            v = s["Виннеры"]
            ratio = f"{v/e:.2f}" if e > 0 else "∞" if v > 0 else "—"
            row = [s["Игрок"], s["Команда"], v,
                   s["Ошибки в сетку"], s["Ошибки в аут"], e, ratio]
            for ci, val in enumerate(row, 1):
                cell = ws.cell(row=ri, column=ci, value=val)
                cell.alignment = Alignment(horizontal="center")
                cell.border = self.BORDER
        for col in ws.columns:
            w = max((len(str(c.value or "")) for c in col), default=10)
            ws.column_dimensions[get_column_letter(col[0].column)].width = w + 3

    # ── Зоны ──────────────────────────────────────────────────────────

    def _zones(self, ws, summaries):
        ws["A1"] = "Время в зонах (сек)"
        ws["A1"].font = self.T_FONT
        znames = list(next(iter(summaries))["zone_time"].keys())
        ws.cell(row=3, column=1, value="Игрок").font = Font(bold=True)
        for c, z in enumerate(znames, 2):
            cell = ws.cell(row=3, column=c, value=z)
            cell.fill = self.H_FILL; cell.font = self.H_FONT
            cell.alignment = Alignment(horizontal="center", wrap_text=True)
        for ri, s in enumerate(summaries, 4):
            ws.cell(row=ri, column=1, value=s["Игрок"])
            for c, z in enumerate(znames, 2):
                ws.cell(row=ri, column=c, value=s["zone_time"].get(z, 0))
        for col in ws.columns:
            ws.column_dimensions[get_column_letter(col[0].column)].width = 20

    # ── Тепловые карты ────────────────────────────────────────────────

    def _heatmap(self, ws, summaries):
        ws["A1"] = "Тепловые карты (5×5) — «своя» сторона всегда снизу"
        ws["A1"].font = self.T_FONT
        oc = 1
        for s in summaries:
            ws.cell(row=3, column=oc,
                    value=f"{s['Игрок']} ({s['Команда']})").font = Font(bold=True)
            hm = s["heatmap"]
            mx = max(max(r) for r in hm) or 1
            for r, row in enumerate(hm):
                for c, val in enumerate(row):
                    cell = ws.cell(row=4+r, column=oc+c, value=val)
                    i = int(255 - (val/mx)*200)
                    cell.fill      = PatternFill("solid", fgColor=f"FF{i:02X}{i:02X}")
                    cell.alignment = Alignment(horizontal="center")
                    cell.border    = self.BORDER
            ws.cell(row=9, column=oc+2, value="↑ сетка").font  = Font(italic=True, size=8)
            ws.cell(row=4, column=oc+2, value="↓ задняя").font = Font(italic=True, size=8)
            oc += 7


# ═══════════════════════════ GUI ════════════════════════════════════════ #

class App(tk.Tk):
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
                 text="Виннеры · Ошибки · Зоны · Тепловые карты",
                 bg="#1A3A5C", fg="#A0C4E8",
                 font=("Helvetica", 10)).pack(pady=(0, 12))

        card = tk.Frame(self, bg="white")
        card.pack(padx=24, pady=4, fill="x")

        # 1. Видео
        self._section(card, "1. Видеофайл")
        r = tk.Frame(card, bg="white"); r.pack(fill="x", padx=12, pady=4)
        tk.Entry(r, textvariable=self.video_path, width=44,
                 font=("Helvetica", 10)).pack(side="left", padx=(0, 8))
        self._btn(r, "Обзор…", self._browse, width=10).pack(side="left")

        # 2. Калибровка
        self._section(card, "2. Калибровка корта (для виннеров/ошибок)")
        cr = tk.Frame(card, bg="white"); cr.pack(fill="x", padx=12, pady=4)
        self._btn(cr, "📐 Калибровать корт", self._calibrate).pack(side="left")
        self.calib_lbl = tk.Label(cr, text="Не задана",
                                   bg="white", fg="#E74C3C",
                                   font=("Helvetica", 10))
        self.calib_lbl.pack(side="left", padx=10)

        # 3. Игроки
        self._section(card, "3. Игроки (до 4)")
        pr = tk.Frame(card, bg="white"); pr.pack(fill="x", padx=12, pady=4)
        self._btn(pr, "+ Добавить игрока", self._add_player).pack(side="left")
        self._btn(pr, "✕ Удалить последнего", self._del_player,
                  color="#E74C3C").pack(side="left", padx=8)
        self.plist = tk.Listbox(card, height=4, font=("Helvetica", 10))
        self.plist.pack(fill="x", padx=12, pady=(0, 8))

        # 4. Лог
        self._section(card, "4. Лог")
        self.log = tk.Text(card, height=7, state="disabled",
                            font=("Courier", 9), bg="#F0F4F8")
        self.log.pack(fill="x", padx=12, pady=(0, 8))
        self.bar = ttk.Progressbar(card, length=420, mode="determinate")
        self.bar.pack(padx=12, pady=(0, 12))

        self._start_btn = tk.Button(
            self, text="▶  Начать анализ", command=self._start,
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
        if width: lbl.config(width=width)
        lbl.pack()
        def on_enter(_): frame.config(bg="#2C5F8A"); lbl.config(bg="#2C5F8A")
        def on_leave(_): frame.config(bg=color);     lbl.config(bg=color)
        def on_click(_): cmd()
        for w in (frame, lbl):
            w.bind("<Enter>",    on_enter)
            w.bind("<Leave>",    on_leave)
            w.bind("<Button-1>", on_click)
        return frame

    def _browse(self):
        p = filedialog.askopenfilename(
            filetypes=[("MP4", "*.mp4"), ("Видео", "*.mp4 *.avi *.mov *.mkv")])
        if p: self.video_path.set(p)

    def _calibrate(self):
        if not self.video_path.get():
            messagebox.showwarning("Ошибка", "Сначала выберите видеофайл!")
            return
        self._log("Открываю калибровку корта…")
        calib = calibrate_court(self.video_path.get(), self)
        if calib:
            self.calib = calib
            self.calib_lbl.config(text="✓ Задана", fg="#27AE60")
            self._log("✓ Калибровка корта сохранена")
        else:
            self._log("Калибровка отменена")

    def _add_player(self):
        if not self.video_path.get():
            messagebox.showwarning("Ошибка", "Сначала выберите видеофайл!")
            return
        if len(self.players) >= 4:
            messagebox.showinfo("Ограничение", "Максимум 4 игрока.")
            return

        dlg = tk.Toplevel(self); dlg.title("Добавить игрока")
        dlg.configure(bg="white"); dlg.grab_set()
        fields = {}
        for label in ("Имя игрока", "Команда"):
            tk.Label(dlg, text=label, bg="white",
                     font=("Helvetica", 10)).pack(padx=20, pady=(10, 2))
            e = tk.Entry(dlg, width=30, font=("Helvetica", 10))
            e.pack(padx=20, pady=(0, 4)); fields[label] = e

        hint = ("Кликните на игрока в окне предпросмотра\n"
                "ПРОБЕЛ/→ — следующий кадр  |  ESC — отмена")
        tk.Label(dlg, text=hint, bg="white", fg="#555",
                 font=("Helvetica", 9), justify="center").pack(padx=20, pady=8)

        def pick():
            name = fields["Имя игрока"].get().strip()
            team = fields["Команда"].get().strip()
            if not name or not team:
                messagebox.showwarning("Ошибка", "Заполните все поля!", parent=dlg)
                return

            already = [p["bbox"] for p in self.players]

            # pick_player_on_frame работает и без YOLO (просто без рамок)
            bbox = pick_player_on_frame(
                self.video_path.get(), name, already, self)

            if bbox is None or bbox[2] == 0 or bbox[3] == 0:
                messagebox.showwarning("Отмена", "Игрок не выбран.", parent=dlg)
                return

            self.players.append({"name": name, "team": team, "bbox": list(bbox)})
            self.plist.insert("end", f"  {name}  [{team}]")
            self._log(f"Добавлен: {name} ({team})")
            dlg.destroy()
            # Автоматически открываем выбор следующего игрока (макс. 4)
            if len(self.players) < 4:
                self.after(150, self._add_player)

        self._btn(dlg, "Выбрать на видео", pick,
                  font=("Helvetica", 11)).pack(pady=12)

    def _del_player(self):
        if self.players:
            p = self.players.pop()
            self.plist.delete("end")
            self._log(f"Удалён: {p['name']}")

    def _start(self):
        if not self.video_path.get():
            messagebox.showwarning("Ошибка", "Выберите видеофайл!"); return
        if not self.players:
            messagebox.showwarning("Ошибка", "Добавьте игроков!"); return
        out = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")])
        if not out: return
        # Сброс прогресс-бара и блокировка кнопки на время анализа
        self.bar["value"] = 0
        self._start_btn.config(state="disabled", bg="#95A5A6",
                               text="⏳  Анализ выполняется…")
        threading.Thread(target=self._run, args=(out,), daemon=True).start()

    def _run(self, out):
        try:
            an = VideoAnalyzer(
                self.video_path.get(), self.players, self.calib,
                progress_cb=lambda p: self.after(0, lambda: self.bar.config(value=p)),
                log_cb=lambda m: self.after(0, self._log, m),
            )
            summaries, sw = an.run()
            ReportGenerator().generate(summaries, self.video_path.get(), sw, out)
            self.after(0, self._done, out)
        except Exception as e:
            self.after(0, self._reset_start_btn)
            self.after(0, messagebox.showerror, "Ошибка", str(e))

    def _reset_start_btn(self):
        self._start_btn.config(state="normal", bg="#27AE60",
                               text="▶  Начать анализ")

    def _done(self, out):
        self.bar["value"] = 100
        self._reset_start_btn()
        self._log(f"📊 Отчёт: {out}")
        if messagebox.askyesno("Готово!", f"Отчёт сохранён:\n{out}\n\nОткрыть?"):
            if os.name == "nt":
                os.startfile(out)
            else:
                subprocess.run(["open", out])  # macOS/Linux — безопасно, без sh-инъекций

    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")


if __name__ == "__main__":
    App().mainloop()