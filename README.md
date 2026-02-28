# 🎾 Padel Match Analyzer

AI-powered desktop application for automatic padel match video analysis.

Padel Match Analyzer uses computer vision and deep learning to automatically
analyse recorded padel matches: it tracks all four players with YOLOv8 +
ByteTrack, detects the ball with a Kalman-filtered frame-difference detector,
classifies every rally-ending event (winner / net error / out error), and
produces a structured Excel report — all from a single video file.

---

## ✨ Features

| Category | Details |
|----------|---------|
| **Ball tracking** | Frame-difference + HSV mask → 97-99 % detection rate; 4D Kalman filter (x, y, vx, vy) for gap-bridging |
| **Player tracking** | YOLOv8n + ByteTrack with automatic ID-swap recovery; CSRT fallback when YOLO unavailable |
| **Event detection** | Winners, net errors, out errors — with time-aware velocity and hitter attribution |
| **Side-switch detection** | Automatically detects when teams swap court ends |
| **Statistics** | Distance, sprints, zone-time heatmap, winners/errors per player |
| **Report** | Multi-sheet Excel workbook: summary table, event log, heatmaps, point-ratio chart |
| **GUI** | Tkinter desktop app with interactive court calibration wizard |
| **Offline** | Runs entirely on-device — no cloud, no internet |

---

## 🛠 Requirements

```bash
pip install opencv-contrib-python numpy openpyxl pillow
pip install ultralytics supervision          # optional — YOLOv8 + ByteTrack
```

> **Python 3.10+** is recommended. YOLOv8 requires PyTorch (installed
> automatically by `ultralytics`). Without YOLOv8 the app falls back to
> OpenCV CSRT trackers, which still works but is less robust.

---

## 🚀 Quick Start

```bash
python main.py
```

1. **Open video** — select your padel match recording.
2. **Add players** — click each player in the first frame and enter name / team.
3. **Calibrate court** — click the four court corners and the two net endpoints.
4. **Analyse** — click *Start Analysis* and wait for the progress bar.
5. **Report** — the Excel file opens automatically when finished.

---

## 🧠 How It Works

```
Video frames
    │
    ├─► YOLOv8n + ByteTrack ──► PlayerTracker (stats, heatmap, zone time)
    │                                   │
    ├─► Frame-diff + HSV mask           │
    │   + Kalman filter ──► BallDetector│
    │                            │      │
    │                            ▼      ▼
    │                       EventDetector
    │                       (winners / errors)
    │
    └─► SideSwitchDetector ──► start_new_period()
                                        │
                                        ▼
                               ReportGenerator (Excel)
```

**Ball detection pipeline**
- Subtract consecutive greyscale frames (`cv2.absdiff`) and apply HSV colour
  mask to isolate the yellow-green ball.
- A 4D Kalman filter (state: x, y, vx, vy) bridges up to 18 missed frames.
- Court-polygon gating rejects candidates outside the court.

**Player tracking**
- YOLOv8n detects all persons every processed frame (~12 fps).
- ByteTrack assigns stable IDs; automatic ID-swap recovery re-matches lost
  players using their last-known pixel position.
- Falls back to OpenCV MIL / CSRT if `ultralytics` / `supervision` are absent.

**Event detection**
- A hit is registered when the ball's velocity vector reverses by > 63° and
  a player is within 250 px of the ball.
- A **winner** requires the ball to cross the net, reach peak speed > 250 px/s
  on the opponent's side, then slow below 70 px/s for ≥ 2 frames.
- A **net error** triggers when the ball stops near the net (< 40 px/s) on the
  hitter's own side within 3 s of the last registered hit.
- An **out error** triggers when the ball leaves the court polygon by > 50 px
  for ≥ 3 consecutive frames.

---

## 📊 Excel Report Sheets

| Sheet | Contents |
|-------|---------|
| **Summary** | Winners, errors, distance, sprints per player — with bar charts |
| **Events** | Timestamped event log (winner / net / out) with hitter name |
| **Heatmaps** | 5 × 5 court coverage grid per player |
| **Point Ratio** | Cumulative (winners − errors) line chart over match time |
| **Side Stats** | Per-period stats when side switches are detected |

---

## 📁 Project Structure

```
main.py          – Full application (GUI + analysis engine)
README.md        – This file
.gitignore
```

---

## 🔧 Configuration (inside main.py)

All tunable constants are class-level attributes:

| Class | Key constants |
|-------|--------------|
| `BallDetector` | `GATE_PX=500`, `MISS_MAX=18`, `DIFF_THR=18` |
| `EventDetector` | `CONTACT_DIST=250`, `WINNER_MIN_SPEED_OPP=250`, `WALL_MARGIN=25` |
| `SideSwitchDetector` | `STABLE_WINDOW=8.0`, `CONFIRM_SEC=1.0`, `COOLDOWN_SEC=90.0` |
| `YOLOByteTracker` | `CONF=0.18`, `BUFFER=60`, `MAX_MATCH_PX=350` |

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Supervision (ByteTrack)](https://github.com/roboflow/supervision)
- [OpenCV](https://opencv.org/)
