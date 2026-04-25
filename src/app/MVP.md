# EyeCursor MVP Specification

## 1. Product Summary

EyeCursor is a cross-platform desktop application that lets users control the mouse cursor using camera-based face and eye input.

The MVP converts the current prototype scripts into one polished desktop app with selectable tracking modes, user profiles, camera setup, calibration flows, and safe cursor-control behavior.

The MVP prioritizes:

- A polished UI/UX suitable for a university demo
- Modularity so new tracking modes can be added later
- Reliable per-user calibration
- Windows support first, then Linux, then macOS
- Safe interaction so the user can pause or stop tracking at any time

---

## 2. MVP Goals

The MVP must allow a user to:

1. Launch one desktop app called **EyeCursor**.
2. Create or select a user profile.
3. Choose one of the supported tracking modes.
4. Discover and select connected cameras.
5. Complete the required calibration for the selected mode.
6. Start, pause, stop, and recalibrate tracking.
7. Use the selected mode to control the cursor.
8. Save calibration data per user and per mode.
9. Reset calibration data when needed.

---

## 3. Supported Tracking Modes

### 3.1 One-Camera Head Pose Mode

Uses one camera to estimate head pose and map head movement to cursor movement.

Includes:

- Camera selection
- Head-pose calibration
- Eye-gesture calibration
- Cursor movement
- Click and scroll gestures
- Per-profile calibration storage

This mode is the first mode that should be made fully working because it has the simplest hardware requirement.

---

### 3.2 Two-Camera Head Pose Mode

Uses two cameras to estimate head pose with stereo information.

Includes:

- Left camera selection
- Right camera selection
- Stereo calibration using a grid/checkerboard
- Head-pose calibration
- Eye-gesture calibration
- Cursor movement
- Click and scroll gestures
- Per-profile calibration storage

Two-camera mode must **not** use hardcoded stereo calibration values.

The following values from the prototype must become dynamic calibration output:

```python
BASELINE_METERS
K1
D1
K2
D2
R
T
```

The app should store these values in a profile-specific stereo calibration file after the user completes stereo calibration.
---

### 3.3 Eye-Gaze-Only Mode

Uses gaze direction to control the cursor.

Includes:

- Camera selection
- Eye-gaze calibration
- Optional model/weights selection if required by the gaze pipeline
- Cursor movement through gaze estimation
- Per-profile calibration storage

Eye-gaze-only mode does **not** require eye gestures for the MVP.

---

## 4. Out of Scope for MVP

The MVP does not include:

- Demo mode
- Fake input simulation
- Mobile support
- Web app support
- Cloud profiles
- Login accounts
- Online sync
- Automatic model downloading unless it is simple and reliable
- Advanced analytics
- Full installer polish beyond PyInstaller builds

---

## 5. Recommended Technology Stack

### 5.1 UI Framework

Use **PySide6 / Qt for Python** for the MVP.

Reasons:

- Better-looking desktop UI than Tkinter
- Supports modern layouts, cards, sidebars, stacked pages, dialogs, and setup wizards
- Cross-platform on Windows, Linux, and macOS
- Better long-term fit as more modes are added
- Still allows reuse of the existing Python tracking code

Tkinter may remain in the old prototype scripts, but the MVP app UI should be PySide6.

---

### 5.2 Packaging

Use **PyInstaller**.

Build separately on each platform:

- Windows executable on Windows
- Linux executable on Linux
- macOS app on macOS

Windows is the top priority.

---

### 5.3 Profile Storage

Use `platformdirs` to store app data in the correct user folder on each OS.

Example:

```python
from platformdirs import user_data_dir

APP_DATA_DIR = user_data_dir("EyeCursor", "EyeCursorTeam")
```

Expected locations:

```text
Windows: C:\Users\<user>\AppData\Local\EyeCursorTeam\EyeCursor
Linux:   ~/.local/share/EyeCursor
macOS:   ~/Library/Application Support/EyeCursor
```

---

## 6. User Profiles

Multiple users may use the same app on the same machine. Each user must have a separate profile.

Each profile stores:

- User display name
- Selected default mode
- Preferred camera for one-camera modes
- Preferred left/right cameras for two-camera mode
- Per-mode calibration data
- Eye-gesture calibration data
- Stereo calibration data
- UI preferences if needed later

Recommended profile folder structure:

```text
EyeCursor/
  profiles/
    user_001/
      profile.json
      calibrations/
        one_camera_head_pose.json
        two_camera_head_pose.json
        eye_gaze.json
        eye_gestures.json
      stereo/
        calibration.json
  app_settings.json
  logs/
```

### 6.1 Profile Features

The app must support:

- Create profile
- Select profile
- Rename profile
- Delete profile
- Reset all calibration for a profile
- Reset calibration for one mode only
- Recalibrate the current mode
- Show whether each mode is calibrated or not calibrated

---

## 7. Calibration System

Calibration is required because users have different faces, eyes, cameras, lighting, screen sizes, and sitting positions.

Calibration must be saved per profile and per mode.

### 7.1 Calibration Rules

- Real cursor control must be disabled during calibration.
- During calibration, show a virtual cursor or target indicator inside the app.
- The user must be able to cancel calibration.
- The user must be able to restart calibration.
- The app must show whether calibration passed or failed.
- The app should show a calibration quality score.

---

### 7.2 Head-Pose Calibration

Used by:

- One-camera head pose mode
- Two-camera head pose mode

Flow:

1. Ask the user to sit normally.
2. Show a center target.
3. Capture neutral head pose.
4. Show targets around the screen area.
5. Ask the user to look or turn slightly toward each target.
6. Estimate yaw/pitch mapping.
7. Save calibration values.

Saved values may include:

- Center yaw
- Center pitch
- Yaw span
- Pitch span
- Smoothing strength
- Screen mapping values
- Calibration quality score
- Date/time created

---

### 7.3 Eye-Gesture Calibration

Used by:

- One-camera head pose mode
- Two-camera head pose mode

The current prototype uses hardcoded thresholds such as:

```python
BOTH_EYES_SQUINT_SCROLL_THRESHOLD
BOTH_EYES_OPEN_SCROLL_THRESHOLD
WINK_EYE_CLOSED_THRESHOLD
WINK_EYE_OPEN_THRESHOLD
```

These should become dynamic per-user calibration values.

Flow:

1. Ask user to open both eyes normally.
2. Capture open-eye ratio range.
3. Ask user to close left eye / left wink.
4. Capture left-wink ratio range.
5. Ask user to close right eye / right wink.
6. Capture right-wink ratio range.
7. Ask user to squint both eyes.
8. Capture both-eyes-squint ratio range.
9. Compute thresholds from user-specific measurements.
10. Save the thresholds in the profile.

Saved values may include:

- Left eye open average
- Right eye open average
- Left wink threshold
- Right wink threshold
- Both eyes open threshold
- Both eyes squint threshold
- Hold duration for click/drag
- Hold duration for scroll
- Scroll amount
- Calibration quality score

---

### 7.4 Eye-Gaze Calibration

Used by:

- Eye-gaze-only mode

Flow:

1. Ask the user to sit normally.
2. Show calibration points on the screen.
3. Ask the user to look at each point.
4. Capture gaze estimates.
5. Build mapping from gaze estimate to screen position.
6. Validate with a few repeated points.
7. Save calibration.

Saved values may include:

- Gaze center
- Gaze-to-screen mapping
- Smoothing values
- Screen bounds
- Model/weights path if needed
- Calibration quality score
- Date/time created

---

### 7.5 Stereo Calibration

Used by:

- Two-camera head pose mode

Two-camera mode must require stereo calibration before tracking starts.

Flow:

1. User selects left and right cameras.
2. App shows live preview from both cameras.
3. User holds a checkerboard/grid in view.
4. App captures multiple valid calibration frames.
5. App computes stereo calibration values.
6. App shows calibration quality.
7. App saves calibration values to the current profile.
8. Two-camera mode becomes available only after successful calibration.

Saved values:

```text
baseline_meters
K1
D1
K2
D2
R
T
image_size
checkerboard_size
square_size
reprojection_error
calibration_date
left_camera_id
right_camera_id
```

Quality indicator:

- Good: low reprojection error and enough valid frames
- Acceptable: usable but may be less stable
- Poor: user should recalibrate

The UI should clearly explain that stereo calibration depends on the selected camera pair. If the user changes either camera, the app should require recalibration.

---

## 8. Calibration Quality Score

Calibration should not just say “done.” It should show quality.

Recommended quality labels:

```text
Excellent
Good
Acceptable
Poor
Failed
```

### 8.1 Head-Pose Quality

Use:

- Face detection stability
- Head pose jitter while looking at center
- Consistency when returning to center
- Range coverage across calibration targets

### 8.2 Eye-Gesture Quality

Use:

- Separation between open-eye and closed-eye ratios
- Consistency across repeated blinks/winks
- Whether left and right wink signals are distinguishable
- Whether squint is clearly different from normal open eyes

### 8.3 Eye-Gaze Quality

Use:

- Average error between target point and predicted point
- Consistency across repeated targets
- Stability while looking at the same point

### 8.4 Stereo Quality

Use:

- Number of valid checkerboard/grid captures
- Reprojection error
- Left/right camera synchronization stability
- Whether calibration was created for the currently selected camera pair

---

## 9. Camera Management

Camera discovery must be part of the app.

The app must allow the user to:

- Scan for cameras
- See live preview cards
- Select one camera for one-camera head pose mode
- Select one camera for eye-gaze mode
- Select left and right cameras for two-camera mode
- Rename cameras locally if useful, for example `Laptop Webcam`, `Left USB Camera`, `Right USB Camera`

### 9.1 Cross-Platform Discovery

The app should not rely only on Linux `/dev/video*`.

Use OpenCV index scanning as the cross-platform baseline:

```text
Try camera indexes 0 through 10
Open each camera
Read one frame
Show successful cameras as preview cards
```

Linux-specific `/dev/video*` scanning can be used as an optimization, but the generic OpenCV scan should exist for Windows and macOS.

### 9.2 Camera Selection Rules

For one-camera modes:

- User selects one camera.
- App stores that camera choice in the profile.

For two-camera mode:

- User selects left camera.
- User selects right camera.
- Left and right cameras cannot be the same camera.
- If either camera changes, stereo calibration becomes invalid until recalibrated.

---

## 10. Eye Gesture Mapping

Eye gestures are used only in the head-pose modes for the MVP.

Default mapping:

| Gesture | Action |
|---|---|
| Left wink | Left click |
| Right wink | Right click |
| Hold left wink | Hold left mouse button / drag |
| Hold right wink | Hold right mouse button |
| Both eyes open and hold | Scroll one direction |
| Both eyes squint and hold | Scroll opposite direction |

The UI must show these mappings clearly during onboarding, calibration, and settings.

Future versions should allow gesture remapping, but the MVP only needs to show the mappings and save calibrated thresholds.

---

## 11. Safety Controls

The app controls the real mouse, so safety is required.

### 11.1 Required Safety Features

- `Q` stops tracking when the EyeCursor app window is focused.
- `Esc` stops tracking when the EyeCursor app window is focused.
- A visible **Pause Tracking** button must always be available while tracking is active.
- A visible **Stop Tracking** button must always be available while tracking is active.
- Real cursor control must be disabled during calibration.
- Cursor control should only start after calibration is complete.
- When tracking starts, show a clear status like `Tracking Active`.
- When tracking is paused, show `Tracking Paused`.

### 11.2 Startup Behavior

When the user presses Start Tracking:

1. App confirms required calibration exists.
2. App opens the selected camera or cameras.
3. App shows preview.
4. App starts tracking.
5. App shows tracking status.

Optional but recommended:

- Short countdown before real cursor movement begins.
- Example: `Starting in 3...2...1`.

---

## 12. UI/UX Requirements

The UI should look polished and modern because presentation quality matters.

### 12.1 Main Layout

Use a modern desktop layout:

```text
Left sidebar:
  Dashboard
  Modes
  Cameras
  Calibration
  Profiles
  Settings

Main content:
  Cards
  Live previews
  Mode setup
  Calibration steps
  Tracking status
```

### 12.2 Dashboard

Dashboard should show:

- Current profile
- Selected mode
- Calibration status for each mode
- Connected cameras
- Big `Start Tracking` button
- Quick actions:
  - Pause
  - Stop
  - Recalibrate
  - Change mode

### 12.3 Mode Selection Page

Show mode cards:

```text
One-Camera Head Pose
Simple setup, one webcam, supports eye gestures.

Two-Camera Head Pose
More advanced, requires stereo calibration.

Eye Gaze
Uses gaze direction, requires gaze calibration.
```

Each card should show:

- Required cameras
- Calibration status
- Start/setup button
- Short explanation

### 12.4 Camera Setup Page

Show camera cards with:

- Live preview
- Camera index
- Resolution
- Select button
- Label field

For two-camera mode:

- Left camera preview
- Right camera preview
- Swap cameras button
- Recalibrate stereo button

### 12.5 Calibration UI

Calibration should use a step-by-step wizard:

```text
Step 1: Camera check
Step 2: Position yourself
Step 3: Capture calibration samples
Step 4: Quality result
Step 5: Save calibration
```

Use:

- Progress indicator
- Clear instructions
- Large targets
- Live preview
- Quality score
- Retry button
- Save button

### 12.6 Tracking Screen

While tracking is active, show:

- Current mode
- Current profile
- Camera status
- Tracking active/paused status
- Gesture status
- Pause button
- Stop button
- Recalibrate button

---

## 13. Technical Architecture

The app should separate UI from tracking logic.

Recommended modules:

```text
src/
  app/
    main.py
    bootstrap.py
  ui/
    main_window.py
    pages/
      dashboard_page.py
      modes_page.py
      cameras_page.py
      calibration_page.py
      profiles_page.py
      settings_page.py
    components/
      camera_preview.py
      mode_card.py
      calibration_score.py
  core/
    modes/
      base.py
      registry.py
      one_camera_head_pose.py
      two_camera_head_pose.py
      eye_gaze.py
    profiles/
      profile_manager.py
      profile_model.py
    calibration/
      calibration_manager.py
      head_pose_calibration.py
      eye_gesture_calibration.py
      gaze_calibration.py
      stereo_calibration.py
    devices/
      camera_manager.py
      camera_model.py
    safety/
      tracking_guard.py
```

---

## 14. Mode Plugin Interface

Every tracking mode should implement the same interface.

Example:

```python
class TrackingMode:
    id: str
    display_name: str
    description: str
    required_camera_count: int
    requires_head_pose_calibration: bool
    requires_eye_gesture_calibration: bool
    requires_stereo_calibration: bool
    requires_gaze_calibration: bool

    def validate_requirements(self, profile, selected_devices) -> bool:
        ...

    def calibrate(self, profile, selected_devices) -> None:
        ...

    def start(self, profile, selected_devices, cursor) -> None:
        ...

    def pause(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def reset_calibration(self, profile) -> None:
        ...
```

Adding a future mode should require:

1. Creating a new mode class.
2. Registering it in the mode registry.
3. Providing required calibration steps.
4. Providing a UI card description.

The main app should not need major rewrites.

---

## 15. Data Models

### 15.1 Profile JSON

Example:

```json
{
  "id": "user_001",
  "display_name": "User 1",
  "default_mode": "one_camera_head_pose",
  "preferred_cameras": {
    "one_camera": 0,
    "eye_gaze": 0,
    "two_camera_left": 1,
    "two_camera_right": 2
  },
  "created_at": "2026-04-25T00:00:00",
  "updated_at": "2026-04-25T00:00:00"
}
```

### 15.2 Calibration Status

Example:

```json
{
  "mode_id": "one_camera_head_pose",
  "is_calibrated": true,
  "quality_label": "Good",
  "quality_score": 0.86,
  "created_at": "2026-04-25T00:00:00"
}
```

### 15.3 Stereo Calibration JSON

Example:

```json
{
  "left_camera_id": 1,
  "right_camera_id": 2,
  "baseline_meters": 0.0788,
  "K1": [[542.97, 0.0, 347.62], [0.0, 542.58, 266.59], [0.0, 0.0, 1.0]],
  "D1": [[0.139, -0.092, 0.013, 0.018, -1.438]],
  "K2": [[550.59, 0.0, 354.94], [0.0, 547.42, 257.20], [0.0, 0.0, 1.0]],
  "D2": [[0.061, 0.096, 0.009, 0.016, -0.325]],
  "R": [[0.999, -0.009, -0.000], [0.009, 0.999, 0.018], [-0.000, -0.018, 0.999]],
  "T": [[-0.078], [-0.000], [0.004]],
  "reprojection_error": 0.42,
  "quality_label": "Good",
  "created_at": "2026-04-25T00:00:00"
}
```

---

## 16. Cross-Platform Requirements

### 16.1 Priority

1. Windows
2. Linux
3. macOS

### 16.2 Windows

Requirements:

- Camera discovery must work through OpenCV.
- Packaged `.exe` should be tested first.
- App should handle camera permission issues gracefully.
- App should avoid Linux-only assumptions.

### 16.3 Linux

Requirements:

- Camera discovery should work through OpenCV index scanning.
- Linux-specific device scanning may be used as an optimization.
- Existing Linux dependencies should be documented.
- Cursor backend dependencies should be checked at startup.

### 16.4 macOS

Requirements:

- Camera permission handling should be documented.
- Accessibility permission may be required for cursor control.
- App should show a clear message if permissions are missing.

---

## 17. Error Handling

The app should show friendly errors for:

- No camera found
- Selected camera unavailable
- Left and right cameras are the same
- Stereo calibration missing
- Stereo calibration invalid after camera change
- Eye-gaze model/weights missing
- Cursor backend failed to initialize
- Camera permission denied
- Calibration failed quality threshold
- Unsupported OS behavior

Error messages should tell the user what to do next.

Bad:

```text
RuntimeError: camera failed
```

Good:

```text
Could not open Camera 1. Try selecting another camera or closing other apps that may be using it.
```

---

## 18. Development Roadmap

### Phase 1: App Shell and Architecture

- Create PySide6 app shell
- Add sidebar navigation
- Add dashboard
- Add mode registry
- Add profile manager
- Add app-data folder support through `platformdirs`

Deliverable:

- App launches
- User can create/select profile
- User can switch between pages
- Modes appear as cards

---

### Phase 2: Camera Discovery

- Implement cross-platform camera scanning
- Show camera preview cards
- Store selected cameras in profile
- Support one-camera and two-camera selection

Deliverable:

- User can scan cameras
- User can choose camera for one-camera modes
- User can choose left/right cameras for two-camera mode

---

### Phase 3: One-Camera Head Pose Mode

- Refactor existing one-camera prototype into mode class
- Connect mode to app UI
- Add head-pose calibration flow
- Add eye-gesture calibration flow
- Add start/pause/stop tracking

Deliverable:

- One-camera mode works from the new app UI

---

### Phase 4: Eye Gesture Calibration

- Replace hardcoded gesture thresholds
- Add per-user threshold calculation
- Show gesture mapping in UI
- Save thresholds in profile

Deliverable:

- Left wink maps to left click
- Right wink maps to right click
- Scroll gestures use calibrated thresholds

---

### Phase 5: Two-Camera Mode and Stereo Calibration

- Refactor two-camera prototype into mode class
- Add stereo calibration wizard
- Save `K1`, `D1`, `K2`, `D2`, `R`, `T`, and baseline dynamically
- Prevent tracking until stereo calibration exists
- Invalidate stereo calibration if selected cameras change

Deliverable:

- Two-camera mode works without hardcoded calibration values

---

### Phase 6: Eye-Gaze Mode

- Refactor eye-gaze prototype into mode class
- Add gaze calibration flow
- Add model/weights selection if needed
- Save gaze calibration in profile

Deliverable:

- Eye-gaze mode launches and uses saved calibration

---

### Phase 7: Packaging and Testing

- Package Windows build first
- Test on Windows
- Package Linux build
- Test on Linux
- Package macOS build if possible
- Document known platform limitations

Deliverable:

- Working Windows build
- Linux build if time allows
- macOS build if time allows

---

## 19. MVP Acceptance Criteria

The MVP is complete when:

1. The app launches as a PySide6 desktop app.
2. User can create and switch profiles.
3. User can discover and select cameras.
4. User can select one of the supported modes.
5. One-camera head pose mode works through the app.
6. Eye-gesture thresholds are calibrated per user.
7. Two-camera mode requires stereo calibration before use.
8. Stereo calibration values are saved dynamically, not hardcoded.
9. Eye-gaze mode has a calibration flow and can be launched from the app.
10. `Q` and `Esc` stop tracking when the app is focused.
11. Pause and Stop buttons are visible while tracking is active.
12. Calibration data is saved in OS-specific user app-data folders.
13. UI is polished enough for a university project presentation.
14. Code structure allows new modes to be added later.

---

## 20. Final MVP Recommendation

Build the MVP as a polished PySide6 desktop application with a modular tracking-mode architecture.

Start with one-camera head pose mode because it is the easiest to stabilize. Then add calibrated eye gestures, camera discovery, two-camera stereo calibration, and eye-gaze mode.

The app should feel like a real product:

- Profiles
- Mode cards
- Camera previews
- Calibration wizards
- Quality scores
- Clear tracking status
- Safe pause/stop controls

The most important technical rule is that prototype constants must move into profile-based calibration files. The most important UX rule is that the user should always understand what mode is active, whether calibration is valid, and how to stop tracking.
