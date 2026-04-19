# AutoScroll — Gesture-Controlled Scrolling, Cursor & Click

A production-grade, system-level application that lets you **scroll**, **move the mouse cursor**, and **click** in **any application** (browser, IDE, PDF viewer, terminal) using real-time hand gestures captured via webcam.

## How It Works

```
Webcam → MediaPipe Hands → Gesture Gate → Mode Dispatch → OS Events
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
                 CURSOR      SCROLL      CLICK
              (move mouse)  (scroll Δy)  (pinch→click)
```

1. **Hand Tracking**: MediaPipe extracts 21 hand landmarks at 30+ FPS
2. **Gesture Classification**: Geometric finger-extension ratios determine the active mode
3. **Mode Dispatch**: Each mode has its own processing pipeline with independent smoothing
4. **OS Events**: `pynput` dispatches native mouse move, scroll, and click events

## Gesture Modes

| Gesture | Fingers | Mode | Action |
|---------|---------|------|--------|
| ☝️ **Index only** | Index extended, others folded | **CURSOR** | Move mouse pointer |
| ✌️ **Peace sign** | Index + middle extended, ring + pinky folded | **SCROLL** | Scroll up/down |
| 🤏 **Pinch** | Thumb tip touches index tip (while in cursor mode) | **CLICK** | Mouse click |
| ✋ **Open palm** | All fingers extended | Idle | No action |
| ✊ **Fist** | All fingers folded | Idle | No action |

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
| **Windows** | Works out of the box. Run as admin to control elevated apps. |
| **macOS** | Grant Accessibility permission: System Preferences → Security & Privacy → Privacy → Accessibility |
| **Linux (X11)** | Works directly. Wayland may require `sudo` or adding user to `input` group. |

## Usage

```bash
# Basic run (scroll + cursor + click)
python main.py

# With calibration phase (recommended for first use)
python main.py --calibrate

# Scroll-only mode (disable cursor/click)
python main.py --no-cursor

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

## Cursor Control

Point with your **index finger only** (fold all other fingers). The mouse cursor follows your fingertip.

- **Active zone**: Only the central 70% of the camera frame maps to the full screen — you don't need large arm movements
- **Smoothing**: Multi-stage pipeline (1€ filter → EMA → dead-zone) eliminates jitter while staying responsive
- **Fine positioning**: Slow movements automatically increase smoothing for pixel-precise control

## Click Detection

While in cursor mode, **pinch your thumb and index finger together** to click.

- **Single click**: Quick pinch and release
- **Double click**: Two rapid pinches within 400ms
- **Hysteresis**: Separate engage (distance < 0.045) and release (distance > 0.065) thresholds prevent accidental oscillation
- **Cooldown**: 250ms minimum between clicks prevents multi-fire
- **Safety**: Clicks are only detected in cursor mode — pinching during scroll mode does nothing

## Scroll Gesture

Make a **"peace sign"**: extend your **index** and **middle** fingers, fold your **ring** and **pinky** fingers.

- **Move hand down** → scroll down
- **Move hand up** → scroll up
- **Slow movement** → precise, fine scrolling
- **Fast movement** → accelerated scrolling

Release the gesture (open palm, fist, etc.) to stop scrolling.

## Architecture

```
main.py                    — Entry point, orchestrator, debug overlay
├── video_capture.py       — Threaded webcam capture (non-blocking)
├── hand_tracker.py        — MediaPipe Hands wrapper (21 landmarks)
├── gesture_detector.py    — Geometric gesture classifier + debounce
├── motion_analyzer.py     — Δy extraction, 1€ filter, dead zone, velocity mapping
├── scroll_controller.py   — OS-level scroll dispatch (pynput/pyautogui)
├── cursor_controller.py   — Camera→screen coordinate mapping + cursor movement
├── click_detector.py      — Pinch detection FSM + click dispatch
└── utils.py               — Filters (EMA, 1€), FPS counter, calibration, screen utils
```

## Key Algorithms

### Gesture Classification
Uses **Euclidean distance ratios** from wrist to fingertip vs. wrist to PIP joint. A finger is "extended" when `dist(tip, wrist) / dist(pip, wrist) > 1.15`. This is rotation-invariant — works regardless of hand orientation.

### Coordinate Mapping (Cursor)
The camera frame is divided into an **active zone** (default: 15–85% X, 10–80% Y). Hand positions within this zone map linearly to the full screen resolution. Positions outside are clamped to screen edges. This allows full-screen cursor control with comfortable wrist-range movements.

### Smoothing Pipeline (Cursor)
```
raw landmark → 1€ filter → screen mapping → EMA → dead-zone gate → OS cursor
```
Velocity-adaptive: slow movements automatically increase smoothing (α=0.20) for precision, while fast movements use default smoothing (α=0.35) for responsiveness.

### Click Detection (Pinch)
Finite state machine with hysteresis:
```
IDLE →(pinch)→ PINCHING →(confirm)→ CLICKED →(cooldown)→ COOLDOWN →(release)→ IDLE
```
Separate engage/release thresholds (0.045 / 0.065) create a hysteresis band that prevents oscillation.

### 1€ Filter (Noise Reduction)
Adaptive low-pass filter that adjusts cutoff frequency based on signal speed:
- **Low speed** → aggressive smoothing (eliminates jitter)
- **High speed** → minimal smoothing (preserves responsiveness)

### Dynamic Scroll Scaling
- Below threshold: **linear** mapping (fine control)
- Above threshold: **power-law** (exponent 1.6) acceleration

### Dead Zone
Movements smaller than the calibrated dead zone are ignored, preventing micro-tremor scrolling or cursor jitter.

## Debug Overlay

The overlay (toggle with `d`) shows:
- **FPS** counter (green ≥25, red <25)
- **Active mode** with color-coded indicator
- **Cursor mode**: screen coordinates, pinch distance bar, click events
- **Scroll mode**: finger Y position, velocity bar
- **Finger markers**: crosshair (cursor), circles (scroll), dim dot (idle)
- **Sensitivity** and dead-zone values

## Calibration

Run `python main.py --calibrate` or press `c` during runtime. Move your hand slowly up and down with the scroll gesture for 5 seconds. The system derives:
- **Dead zone**: 25th percentile of observed movement magnitudes
- **Slow threshold**: Median movement magnitude
- **Sensitivity**: Inverse of 90th percentile

## Performance

- Target: **≥25 FPS** on 640×480 input
- End-to-end latency: **<50 ms** (MediaPipe lite model + threaded capture)
- Cursor movement adds ~0.1ms overhead per frame
- Click detection is pure math on existing landmarks — zero extra cost
- Frame acquisition is non-blocking (dedicated reader thread)
