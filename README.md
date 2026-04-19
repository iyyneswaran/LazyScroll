# AutoScroll — Gesture-Controlled System Scrolling

A production-grade, system-level application that lets you scroll **any application** (browser, IDE, PDF viewer, terminal) using real-time hand gestures captured via webcam.

## How It Works

```
Webcam → MediaPipe Hands → Gesture Gate → Motion Δy → Filter → Scroll Events → OS
```

1. **Hand Tracking**: MediaPipe extracts 21 hand landmarks at 30+ FPS
2. **Gesture Gate**: Scrolling activates **only** when index + middle fingers are extended (others folded)
3. **Motion Analysis**: Vertical displacement of the index fingertip is filtered through a 1€ filter + EMA
4. **Scroll Dispatch**: Filtered velocity is mapped to OS-level scroll events via `pynput`

## Setup

### Prerequisites
- Python 3.10+
- Webcam
- Windows / macOS / Linux

### Install

```bash
cd AutoScroll
pip install -r requirements.txt
```

### Platform Notes

| Platform | Notes |
|----------|-------|
| **Windows** | Works out of the box. Run as admin to scroll in elevated apps. |
| **macOS** | Grant Accessibility permission: System Preferences → Security & Privacy → Privacy → Accessibility |
| **Linux (X11)** | Works directly. Wayland may require `sudo` or adding user to `input` group. |

## Usage

```bash
# Basic run
python main.py

# With calibration phase (recommended for first use)
python main.py --calibrate

# Natural (inverted) scrolling
python main.py --invert

# Use a specific camera
python main.py --camera 1

# Custom resolution
python main.py --width 1280 --height 720

# Disable debug overlay
python main.py --no-overlay

# Adjust sensitivity
python main.py --sensitivity 1.5
```

### Keyboard Controls (while running)

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `c` | Toggle calibration mode |
| `d` | Toggle debug overlay |
| `+` / `=` | Increase sensitivity |
| `-` | Decrease sensitivity |

## Scroll Gesture

Make a **"peace sign"**: extend your **index** and **middle** fingers, fold your **ring** and **pinky** fingers. Thumb position doesn't matter.

- **Move hand down** → scroll down
- **Move hand up** → scroll up
- **Slow movement** → precise, fine scrolling
- **Fast movement** → accelerated scrolling

Release the gesture (open palm, fist, etc.) to stop scrolling.

## Architecture

```
main.py                 — Entry point, orchestrator, debug overlay
├── video_capture.py    — Threaded webcam capture (non-blocking)
├── hand_tracker.py     — MediaPipe Hands wrapper (21 landmarks)
├── gesture_detector.py — Geometric gesture classifier + debounce
├── motion_analyzer.py  — Δy extraction, 1€ filter, dead zone, velocity mapping
├── scroll_controller.py— OS-level scroll dispatch (pynput/pyautogui)
└── utils.py            — Filters (EMA, 1€), FPS counter, calibration store
```

## Key Algorithms

### Gesture Classification
Uses **Euclidean distance ratios** from wrist to fingertip vs. wrist to PIP joint. A finger is "extended" when `dist(tip, wrist) / dist(pip, wrist) > 1.15`. This is rotation-invariant — works regardless of hand orientation.

### 1€ Filter (Noise Reduction)
Adaptive low-pass filter that adjusts cutoff frequency based on signal speed:
- **Low speed** → aggressive smoothing (eliminates jitter)
- **High speed** → minimal smoothing (preserves responsiveness)

### Dynamic Scroll Scaling
- Below threshold: **linear** mapping (fine control)
- Above threshold: **power-law** (exponent 1.6) acceleration

### Dead Zone
Movements smaller than the calibrated dead zone are ignored, preventing micro-tremor scrolling.

## Calibration

Run `python main.py --calibrate` or press `c` during runtime. Move your hand slowly up and down with the scroll gesture for 5 seconds. The system derives:
- **Dead zone**: 25th percentile of observed movement magnitudes
- **Slow threshold**: Median movement magnitude
- **Sensitivity**: Inverse of 90th percentile

## Performance

- Target: **≥25 FPS** on 640×480 input
- End-to-end latency: **<50 ms** (MediaPipe lite model + threaded capture)
- Frame acquisition is non-blocking (dedicated reader thread)
- MediaPipe runs with `model_complexity=0` (fastest inference)
