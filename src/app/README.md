# EyeCursor

<p align="center">
  <img src="../../assets/icon_256.png" alt="EyeCursor" width="128">
</p>

<p align="center">
  <b>Control your mouse cursor with head and eye movements.</b>
</p>

EyeCursor is a cross-platform desktop application that lets users control the mouse cursor using camera-based face and eye input. It supports multiple tracking modes, per-user calibration profiles, and click/scroll gestures via facial blendshapes (lip pucker, lip tuck, and smirks).

---

## Features

- **One-Camera Head Pose Mode** -- Move the cursor by turning your head using a single webcam.
- **Two-Camera Stereo Mode** -- Stereo depth-enhanced head tracking with two webcams.
- **Eye-Gaze Mode** -- Control the cursor by looking at screen targets using gaze estimation (ETH-XGaze).
- **Facial Gesture Controls** -- Pucker lips presses & holds the LEFT mouse button; tucking lips inward presses & holds the RIGHT mouse button (brief = single click, sustained past 0.5 s = drag/draw). Smirk LEFT scrolls up; smirk RIGHT scrolls down. Click gestures (lips) are robust while the head is turned; scroll gestures (smirks) are stable when looking straight at the screen. Pucker intensity uses MediaPipe's `mouthPucker` blendshape (we deliberately do not use `cheekPuff`, which is unreliable in MediaPipe's default model); lip tuck uses `max(mouthRollUpper, mouthRollLower, mouthPressLeft, mouthPressRight)`; smirks use `mouthSmileLeft` / `mouthSmileRight`.
- **User Profiles** -- Each user gets their own calibration data. Multiple users can share one machine.
- **Calibration Wizards** -- Step-by-step guided calibration for head pose, facial gestures, stereo cameras, and gaze.
- **Camera Discovery** -- Automatically detect and preview connected cameras.
- **Safety Controls** -- Start/pause/stop tracking anytime. Press Q or Esc to stop immediately.

---

## Supported Platforms

| Platform | Status |
|----------|--------|
| Linux    | Fully supported |
| Windows  | Fully supported |
| macOS    | Supported (requires pyobjc dependencies) |

---

## Installation

### Prerequisites

- **Python 3.10 - 3.12** (3.12 recommended)
- **A webcam** (one for head-pose and gaze modes, two for stereo mode)
- **Git**

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3-dev python3-venv cmake libgl1-mesa-glx libglib2.0-0 \
    libxcb-xinerama0 libxkbcommon-x11-0 libdbus-1-3 libxcb-cursor0 \
    xdotool x11-xserver-utils
```

#### Windows

1. Install Python 3.12 from [python.org](https://www.python.org/downloads/) -- check **"Add to PATH"** during install.
2. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) -- select "Desktop development with C++" (needed for dlib).
3. Install [CMake](https://cmake.org/download/) and add it to PATH.

#### macOS

```bash
xcode-select --install
brew install cmake python@3.12
```

---

### Step 1: Clone the repository

```bash
git clone https://github.com/ItsAboudTime/EyeCursor.git
cd EyeCursor
```

### Step 2: Create a virtual environment

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### Step 3: Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** Installing `dlib` and `torch` may take several minutes. On Windows, if `dlib` fails, make sure CMake and Visual Studio Build Tools are installed.

### Step 4: Download required model files

EyeCursor needs the following model files for Eye-Gaze mode (not included in the repository):

| File | Description |
|------|-------------|
| `shape_predictor_68_face_landmarks.dat` | dlib face landmark detector ([download](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)) |
| `face_model.txt` | 3D face model (included in the ETH-XGaze repository) |
| ETH-XGaze weights (e.g. `epoch_24_ckpt.pth.tar`) | Trained gaze estimation model |

> Head-pose and stereo modes use MediaPipe, which downloads its models automatically on first run.
>
> Eye-gaze mode will prompt you to browse for the weights file during calibration.

---

## Running the App

From the project root directory:

**Linux / macOS:**

```bash
source venv/bin/activate
python -m src.app.main
```

**Windows:**

```powershell
.\venv\Scripts\Activate.ps1
python -m src.app.main
```

### Desktop Shortcut (Linux)

A desktop launcher is provided at the project root (`EyeCursor.desktop`). Either:

- Double-click it from the desktop (right-click > **Allow Launching** if prompted), or
- Copy it to your applications menu:

```bash
cp EyeCursor.desktop ~/.local/share/applications/
```

---

## Usage Guide

### 1. Create a Profile

A "Default User" profile is created automatically on first launch. To add more:

1. Click **Profiles** in the sidebar.
2. Click **Create** and enter a display name.
3. Click **Switch To** to activate the new profile.

Each profile stores its own calibration data independently.

### 2. Select a Camera

1. Click **Cameras** in the sidebar.
2. Click **Scan Cameras** to detect connected webcams.
3. Click **Select** on the camera you want for one-camera modes.
4. For stereo mode, use **Set as Left** and **Set as Right** to assign both cameras.

### 3. Choose a Tracking Mode

1. Click **Modes** in the sidebar.
2. Select one of the three modes:
   - **One-Camera Head Pose** -- simplest setup, one webcam.
   - **Two-Camera Head Pose** -- two webcams + stereo calibration.
   - **Eye Gaze** -- one webcam + ETH-XGaze model weights.

### 4. Calibrate

1. Click **Calibration** in the sidebar.
2. Complete the required calibrations for your chosen mode:

| Mode | Required Calibrations |
|------|-----------------------|
| One-Camera Head Pose | Head Pose + Facial Gestures |
| Two-Camera Head Pose | Head Pose + Facial Gestures + Stereo |
| Eye Gaze | Gaze |

#### Head Pose Calibration

Look at 9 on-screen targets and press **Capture** at each position. This maps your head movement range to the full screen.

#### Facial Gesture Calibration

Follow 5 prompts: relax your face, smirk left, smirk right, pucker your lips outward as fully as is comfortable, and tuck your lips inward (roll/press them together). This sets personalized thresholds for click and scroll gestures based on your natural blendshape range.

#### Stereo Calibration (Two-Camera Mode only)

Hold a checkerboard pattern in front of both cameras. Capture at least 15 frame pairs from different angles. The app computes the stereo geometry between your two cameras.

#### Gaze Calibration (Eye-Gaze Mode only)

Browse for the ETH-XGaze model weights file, then look at 9 on-screen targets and press **Capture** at each. The app fits a mapping from your gaze direction to screen coordinates.

### 5. Start Tracking

1. Go to the **Dashboard**.
2. Click **Start Tracking**.
3. A 3-second countdown gives you time to position yourself.
4. Move your head (or eyes in gaze mode) to control the cursor.

### 6. Gestures (Head Pose Modes)

| Gesture | Action |
|---------|--------|
| Pucker lips (lips outward, as if blowing) | Press & hold LEFT mouse button. Brief = single click; sustained past 0.5 s = drag/draw |
| Tuck lips inward (rolled in or pressed together) | Press & hold RIGHT mouse button (same hold/drag semantics) |
| Smirk LEFT (raise the LEFT side of your mouth) | Scroll up (speed proportional to intensity) |
| Smirk RIGHT (raise the RIGHT side of your mouth) | Scroll down (speed proportional to intensity) |

Cursor is briefly frozen during the first 0.5 s of a smirk, then unfreezes so you can drag while the button is held -- this is what enables drawing in paint apps.

A 200 ms intent buffer on each scroll direction smooths out brief, transient lip motions so they don't fire stray scroll bursts.

### 7. Stop Tracking

- Press **Q** or **Esc** on your keyboard, or
- Click the **Stop** button on the Dashboard.

---

## Calibration Tips

- **Lighting** -- Use even, front-facing lighting. Avoid backlighting or strong shadows on your face.
- **Distance** -- Sit approximately 50-70 cm from the camera.
- **Stability** -- Keep your head still during each calibration capture.
- **Quality scores** -- After each calibration, a quality score is shown. "Good" or "Excellent" means you're ready. "Poor" means you should retry.
- **Recalibrate** -- If tracking feels off, go to Calibration and redo the relevant step.

---

## Settings

Click **Settings** in the sidebar to adjust:

- **Cursor Speed** -- How fast the cursor moves relative to head/gaze movement.
- **Frame Rate** -- Camera capture rate (higher = smoother but more CPU).
- **Scroll Speed** -- Default scroll speed cap (units per second) for smirk-driven scrolls.
- **EMA Smoothing** -- Smoothing factor for cursor movement (lower = smoother, higher = more responsive).

---

## Troubleshooting

**Camera not detected:**
- Make sure your webcam is plugged in and not used by another app.
- On Linux, check permissions: `ls -la /dev/video*`. Your user should be in the `video` group.

**MediaPipe errors on first run:**
- MediaPipe downloads model files automatically. Ensure you have internet access on first launch.

**dlib won't install on Windows:**
- Install CMake and Visual Studio Build Tools with C++ support, then retry `pip install dlib`.

**PySide6 errors on Linux:**
- Install the system libraries listed in the Linux prerequisites section above.

**Tracking feels laggy:**
- Lower the frame rate in Settings.
- Close other apps using the camera.
- Ensure adequate lighting.

**Facial gestures not registering:**
- Redo facial gesture calibration. Capture distinct, exaggerated samples (clear left/right smirks, full pucker, full lip-tuck) so the per-user thresholds land in a reachable range.
- Glasses, facial hair, or strong sidelight on one cheek can depress blendshape scores -- adjust lighting and recalibrate.
