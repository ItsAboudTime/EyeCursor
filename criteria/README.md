# EyeCursor TestLab

<p align="center">
  <img src="../assets/icon_256.png" alt="EyeCursor TestLab" width="128">
</p>

<p align="center">
  <b>Fullscreen cursor-control testing for EyeCursor and other input methods.</b>
</p>

EyeCursor TestLab is a standalone PySide6 desktop application for measuring cursor movement, accuracy, tracking, and clicking performance. It is independent from the main EyeCursor tracking system: it does not access cameras, start tracking pipelines, or control the cursor. It only observes normal cursor position and mouse clicks during repeatable fullscreen tests.

---

## Features

- **Session Setup** -- Participant/session name, input method label, seed, screen size, and notes.
- **Seeded Tests** -- Same seed and screen configuration generate repeatable target sequences.
- **Movement Task** -- Varied circle sizes, 100 ms dwell rule, 3 second timeout, completion and speed metrics.
- **Accuracy Task** -- Small same-size targets, 1 second timing window, pixel and normalized error metrics.
- **Tracking Task** -- Seeded moving target path, regular cursor sampling, average and stability metrics.
- **Clicking Task** -- Left/right required clicks, success, wrong-click-inside, outside-click, and timeout logging.
- **Scoring** -- Final 0-100 score with Excellent, Good, Acceptable, Poor, or Failed rating.
- **Exports** -- Raw JSON session data and CSV summaries for spreadsheet analysis.

---

## Supported Platforms

| Platform | Status |
|----------|--------|
| Linux    | Supported |
| Windows  | Supported through Python/PySide6 |
| macOS    | Supported through Python/PySide6 |

> macOS and Linux desktop security settings can affect fullscreen input/cursor behavior. See the platform notes below.

---

## Installation

### Prerequisites

- **Python 3.10 - 3.12** (3.12 recommended)
- **Git**
- A normal mouse or cursor-control method to test

No webcam or EyeCursor model files are required for TestLab.

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3-dev python3-venv libgl1-mesa-glx libglib2.0-0 \
    libxcb-xinerama0 libxkbcommon-x11-0 libdbus-1-3 libxcb-cursor0
```

#### Windows

1. Install Python 3.12 from [python.org](https://www.python.org/downloads/) and check **"Add to PATH"**.
2. Install Git.
3. Use PowerShell or Command Prompt for the setup commands below.

#### macOS

Install Apple command line tools:

```bash
xcode-select --install
```

If Python 3 is not already installed, install it from [python.org](https://www.python.org/downloads/macos/) or Homebrew:

```bash
brew install python@3.12
```

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/ItsAboudTime/EyeCursor.git
cd EyeCursor
```

If you received the project as a ZIP, extract it and open a terminal in the extracted project folder.

---

## Step 2: Create a Virtual Environment

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

---

## Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** The full project requirements include dependencies for the main EyeCursor tracking app, so installation can take a while. TestLab itself mainly needs PySide6 and platformdirs.

---

## Running TestLab

From the project root directory:

**Linux / macOS:**

```bash
source venv/bin/activate
python -m criteria.app.main
```

Or use the launcher:

```bash
./launch_criteria.sh
```

**Windows:**

```powershell
.\venv\Scripts\Activate.ps1
python -m criteria.app.main
```

### Desktop Shortcut (Linux)

A local desktop launcher can point to `launch_criteria.sh`.

If a shortcut is on the desktop and your desktop environment blocks it, right-click it and choose **Allow Launching** or **Trust and Launch**.

---

## macOS Notes

For a friend trying it on macOS, the shortest path is:

```bash
git clone https://github.com/ItsAboudTime/EyeCursor.git
cd EyeCursor
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m criteria.app.main
```

If macOS shows privacy prompts, allow Terminal or Python access as needed in:

```text
System Settings > Privacy & Security
```

Useful permissions to check:

- **Accessibility** -- can affect observing/using input behavior.
- **Input Monitoring** -- may be needed on some setups for input-related behavior.

The MVP is not packaged as a `.app` yet, so running from Terminal is recommended.

---

## Usage Guide

### 1. Start a Session

1. Open **New Session**.
2. Enter the participant/session name.
3. Choose the input method label:
   - Mouse
   - One-Camera Head Pose
   - Two-Camera Head Pose
   - Eye-Gaze Only
   - Custom
4. Enter or keep the generated seed.
5. Add notes if needed.
6. Click **Start Fullscreen Test**.

### 2. Complete the Tasks

The tests run in this order:

1. Movement
2. Accuracy
3. Tracking
4. Clicking
5. Final Results

Between tasks, use the transition screen to continue, pause, end the session, or view current results.

### 3. Controls

| Control | Action |
|---------|--------|
| `Esc` | Pause the current active task |
| `Q` | Stop the current active task and end the session |
| Continue | Start the next task or resume a paused active task |
| Pause Session | Save and close the fullscreen test window |
| End Session | Mark the session stopped |
| View Results | Return to the results page |

### 4. Resume a Session

1. Open **Resume Session**.
2. Select an incomplete session.
3. Click **Continue Selected Session**.

Stopped or paused sessions continue from the next incomplete task.

---

## Results and Exports

Open **Results** to view:

- Final score and rating
- Participant name
- Input method
- Seed
- Screen size
- Task scores
- Session status

Use:

- **Export JSON** for full raw session data.
- **Export CSV Summary** for spreadsheet-friendly summary data.

---

## Data Storage

Sessions are stored with `platformdirs` in the local EyeCursor TestLab data directory.

Typical locations:

| Platform | Location |
|----------|----------|
| Linux | `~/.local/share/EyeCursor TestLab` |
| Windows | `%LOCALAPPDATA%\EyeCursorTeam\EyeCursor TestLab` |
| macOS | `~/Library/Application Support/EyeCursor TestLab` |

Session folders use this shape:

```text
sessions/<session_id>/
  session.json
  summary.json
  raw_events.json
  movement_trials.csv
  accuracy_trials.csv
  tracking_samples.csv
  clicking_trials.csv
exports/
```

JSON is the source of truth. CSV files are generated for convenience.

---

## Scoring

The final score is weighted:

| Task | Weight |
|------|--------|
| Movement | 30% |
| Accuracy | 30% |
| Tracking | 25% |
| Clicking | 15% |

Ratings:

| Score | Rating |
|-------|--------|
| 90-100 | Excellent |
| 75-89 | Good |
| 60-74 | Acceptable |
| 40-59 | Poor |
| 0-39 | Failed |

---

## MVP Limitations

- The first MVP tests mouse/cursor behavior only.
- The app does not launch or integrate with EyeCursor tracking modes.
- The app does not access cameras or calibration data.
- Pausing during an active task resumes in memory during the same fullscreen run.
- Stopping and later continuing resumes from the next incomplete task, not the middle of an incomplete task.
- Task counts and timings are MVP defaults and are not yet editable in the UI.
- Linux cursor sampling may depend on X11/Wayland behavior and desktop security settings.
- macOS may require privacy permissions for reliable input behavior.
- The app is not packaged as a Windows `.exe` or macOS `.app` yet.

---

## Troubleshooting

**PySide6 import error:**

Make sure the virtual environment is active and dependencies are installed:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

**macOS blocks input behavior:**

Open **System Settings > Privacy & Security** and check Accessibility/Input Monitoring permissions for Terminal or Python.

**Linux shortcut does not open:**

Right-click the desktop shortcut and choose **Allow Launching** or run directly:

```bash
./launch_criteria.sh
```

**Results are not visible:**

Open the app's local data folder listed in **Settings**. Session JSON and CSV files are saved after each completed task.
