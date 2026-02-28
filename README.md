# 🎾 Padel Match Analyzer

AI-powered desktop application for automatic padel match analysis from video.

Padel Match Analyzer is a Python-based desktop application that uses computer vision to automatically analyze recorded padel matches. The system tracks players and the ball, detects match events such as winners and errors, identifies side switches, and generates detailed Excel reports with player statistics.

---

## 🚀 Features

- 🎥 Video-based match analysis
- 👤 Player detection and tracking (Computer Vision / OpenCV)
- 🎾 Ball detection and tracking
- 🧠 Automatic event recognition:
  - Winners
  - Unforced errors
  - Rally endings
  - Side switches
- 📊 Player-based statistical analysis
- 📈 Automatic Excel report generation
- 🖥 Fully offline desktop application

---

## 🛠 Tech Stack

- Python 3.x
- OpenCV
- NumPy
- Pandas
- OpenPyXL / XlsxWriter (for Excel export)
- Desktop GUI (PyQt / Tkinter / Custom UI)

---

## 🧠 How It Works

1. Load a recorded padel match video.
2. The system processes the video frame-by-frame:
   - Detects and tracks players
   - Detects and tracks the ball
   - Identifies rally-ending events
3. Events are classified (winner, error, etc.).
4. Match statistics are calculated per player.
5. A structured Excel report is generated automatically.

---

## 📊 Generated Statistics

The Excel report includes:

- Total points won
- Winners
- Unforced errors
- Rally statistics
- Side-based performance metrics
- Match summary overview

---

## 📦 Installation

```bash
git clone https://github.com/your-username/padel-match-analyzer.git
cd padel-match-analyzer
pip install -r requirements.txt
