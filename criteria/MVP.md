# EyeCursor TestLab MVP Specification

## 1. Product Summary

**EyeCursor TestLab** is a cross-platform fullscreen desktop application used to measure the functional performance of cursor-control systems.

The MVP is designed to test and compare:

- Normal mouse usage
- EyeCursor head-tracking usage
- EyeCursor eye-gaze usage
- Future cursor-control methods

The app should run independently from the main EyeCursor application, but it should visually feel like part of the same product family. Since EyeCursor already uses PySide6, EyeCursor TestLab should also use PySide6 and follow a similar UI style, layout language, and interaction design.

The purpose of the testing app is to provide repeatable, fair, measurable tests for cursor movement, accuracy, tracking, and clicking.

The MVP prioritizes:

- Fullscreen testing
- Repeatable seeded task generation
- A polished PySide6 UI
- Sequential test sessions
- Ability to pause, stop, and continue
- Raw data storage
- Summary metrics
- A final 0-100 performance score
- Exportable results for later analysis

---

## 2. MVP Goals

The MVP must allow a user to:

1. Launch one desktop app called **EyeCursor TestLab**.
2. Start a new test session.
3. Select or enter the test participant name.
4. Select the input method being tested.
5. Configure the seed and task settings.
6. Run the test in fullscreen mode.
7. Complete the four test tasks one after another.
8. Pause between tests.
9. Stop a session and continue later.
10. Save raw test data after each task.
11. View task-level results.
12. View a final score based on all completed tests.
13. Export results as JSON and CSV.
14. Compare different input methods using repeatable test conditions.

---

## 3. Supported Test Inputs

The testing program should not control how the cursor is moved. It should only observe cursor position, cursor movement, and clicks.

The MVP should support labeling a session with one of the following input methods:

### 3.1 Mouse

Used as the baseline comparison.

The mouse input method is useful because it provides a reference for how fast and accurate normal cursor usage is.

### 3.2 One-Camera Head Pose

Used when testing EyeCursor's one-camera head-pose mode.

### 3.3 Two-Camera Head Pose

Used when testing EyeCursor's stereo head-pose mode.

### 3.4 Eye-Gaze Only

Used when testing gaze-based cursor movement.

### 3.5 Other / Custom

Used for future cursor-control methods.

The user should be able to enter a custom label if the tested method is not listed.

---

## 4. Out of Scope for MVP

The MVP does not include:

- Integration with the main EyeCursor tracking pipeline
- Automatic launching of EyeCursor
- Camera access
- Eye tracking
- Head tracking
- Gesture detection
- Cloud sync
- Online accounts
- Web dashboard
- Advanced analytics
- Multi-participant database management
- Statistical significance testing
- Automatic comparison reports across many users
- Installer polish beyond basic PyInstaller builds

The testing app should remain independent from the cursor-control app. This makes the tests fair and allows the same test program to evaluate other input methods later.

---

## 5. Recommended Technology Stack

### 5.1 UI Framework

Use **PySide6 / Qt for Python**.

Reasons:

- EyeCursor already uses PySide6.
- The testing app should match the feel and look of the main app.
- PySide6 supports fullscreen windows, custom drawing, timers, keyboard events, and mouse events.
- It is cross-platform across Windows, Linux, and macOS.
- It supports polished layouts, cards, sidebars, stacked pages, dialogs, and result screens.

### 5.2 Drawing and Test Area

Use a custom `QWidget` or `QGraphicsView` for the test canvas.

Recommended MVP choice:

```text
Custom QWidget with QPainter
```

Reasons:

- Simple to draw circles, targets, progress indicators, and moving objects.
- Easy to handle fullscreen rendering.
- Easy to sample cursor position with `QCursor.pos()`.
- Easy to process mouse clicks and keyboard shortcuts.

### 5.3 Timing

Use:

```text
QElapsedTimer
QTimer
```

`QElapsedTimer` should be used for accurate task timing.

`QTimer` should be used for sampling cursor position during tracking tasks and checking dwell-time conditions.

### 5.4 Data Storage

Use `platformdirs` for cross-platform app data folders.

Example:

```python
from platformdirs import user_data_dir

APP_DATA_DIR = user_data_dir("EyeCursor TestLab", "EyeCursorTeam")
```

Expected locations:

```text
Windows: C:\Users\<user>\AppData\Local\EyeCursorTeam\EyeCursor TestLab
Linux:   ~/.local/share/EyeCursor TestLab
macOS:   ~/Library/Application Support/EyeCursor TestLab
```

### 5.5 Export Formats

The MVP should export:

- JSON for full raw data
- CSV for spreadsheet analysis

JSON should be treated as the source of truth.

CSV should be generated for convenience.

### 5.6 Packaging

Use **PyInstaller**.

Build separately on each platform:

- Windows executable on Windows
- Linux executable on Linux
- macOS app on macOS

Windows should be the first priority.

---

## 6. Test Session Flow

The app should use a session-based flow.

A session represents one complete testing run for one participant using one input method.

### 6.1 Session Setup

Before starting, the user should enter or select:

- Participant name or ID
- Input method
- Test seed
- Screen information
- Task configuration preset
- Notes, if needed

Example input method labels:

```text
Mouse
One-Camera Head Pose
Two-Camera Head Pose
Eye-Gaze Only
Custom
```

### 6.2 Fullscreen Test Mode

All active tests must run fullscreen.

Reasons:

- Cursor-control systems use the whole screen.
- Fullscreen makes results more consistent.
- Windowed testing could distort target positioning and movement distance.

### 6.3 Sequential Task Flow

The MVP should run the tests in this order:

1. Movement Task
2. Accuracy Task
3. Tracking Task
4. Clicking Task
5. Final Results

Between tasks, the app should show a transition screen.

Example:

```text
Movement Task Complete

Continue to Accuracy Task
Pause Session
End Session
View Current Results
```

### 6.4 Stop and Continue

The user should be able to stop after any task and continue later.

The app should save session state after:

- Session creation
- Each completed task
- Each paused task
- Each stopped task
- Final result generation

A resumed session should continue from the next incomplete task.

### 6.5 Emergency Controls

The app should support:

```text
Esc = pause current task or return to transition screen
Q = stop current task
```

A visible stop option should be available on transition and pause screens.

During active tests, the UI should stay minimal, but safety controls must still be available through keyboard shortcuts.

---

## 7. Randomness and Repeatability

Each session must use a seed.

The seed controls:

- Target positions
- Target sizes
- Target order
- Tracking target path
- Click target types

Using the same seed and task configuration should generate the same test sequence.

This is important because it allows fair comparison between:

- Mouse control
- Eye/head tracking
- Different EyeCursor versions
- Different calibration settings
- Different participants

### 7.1 Seed Rules

The MVP should support:

- Auto-generated random seed
- Manual seed entry
- Displaying the active seed
- Saving the seed in session results
- Reusing a seed from an old session

---

## 8. Functional Test Tasks

The MVP includes four tasks:

1. Movement Task
2. Accuracy Task
3. Tracking Task
4. Clicking Task

Each task should be self-contained and should save raw trial data.

---

## 9. Movement Task

### 9.1 Purpose

The movement task measures how quickly the user can move the cursor to visible targets.

### 9.2 Behavior

A blank fullscreen test area appears. A circle appears at a seeded random position. The user must move the cursor into the circle and keep it inside for a short dwell period.

After the target is completed, another circle appears.

### 9.3 Dwell Rule

A target is completed only when the cursor remains inside the circle for:

```text
100 ms
```

This prevents a target from being counted when the cursor only briefly crosses it.

### 9.4 Target Sizes

The movement task should use multiple target sizes:

```text
Large targets: easier, useful for rough movement
Medium targets: normal test difficulty
Small targets: useful for precision movement
```

The exact MVP defaults can be:

```text
Large radius: 60 px
Medium radius: 35 px
Small radius: 20 px
```

These values should be configurable later.

### 9.5 Timeout

Each target should have a timeout.

Recommended MVP default:

```text
3 seconds per target
```

If the timeout is reached, the trial is marked as incomplete and the next target appears.

### 9.6 Final Outputs

The movement task should output:

- Completion count
- Timeout count
- Average movement time
- Median movement time
- Average target distance
- Completion rate
- Movement score

### 9.7 Raw Data Per Trial

Example:

```json
{
  "task": "movement",
  "trial_index": 4,
  "target_x": 820,
  "target_y": 410,
  "target_radius": 35,
  "start_time_ms": 12000,
  "end_time_ms": 12740,
  "movement_time_ms": 740,
  "dwell_time_required_ms": 100,
  "completed": true,
  "timed_out": false
}
```

---

## 10. Accuracy Task

### 10.1 Purpose

The accuracy task measures how close the cursor gets to a target center under a fixed time constraint.

### 10.2 Behavior

A small circle appears at a seeded random position. The user has a fixed amount of time to move the cursor as close as possible to the center of the circle.

At the end of the time window, the app records the cursor position and calculates the error.

### 10.3 Target Size

All accuracy targets should use the same small radius.

Recommended MVP default:

```text
Radius: 20 px
```

### 10.4 Timeout

Recommended MVP default:

```text
1 second per target
```

### 10.5 Error Metrics

The app should store three error values:

```text
pixel_error = distance between cursor and target center
radius_normalized_error = pixel_error / target_radius
screen_normalized_error = pixel_error / screen_diagonal
```

The final accuracy score should mainly use `radius_normalized_error`.

### 10.6 Final Outputs

The accuracy task should output:

- Average pixel error
- Median pixel error
- Average radius-normalized error
- Median radius-normalized error
- Average screen-normalized error
- Accuracy score

### 10.7 Raw Data Per Trial

Example:

```json
{
  "task": "accuracy",
  "trial_index": 7,
  "target_x": 1020,
  "target_y": 530,
  "target_radius": 20,
  "cursor_x": 1006,
  "cursor_y": 522,
  "pixel_error": 16.12,
  "radius_normalized_error": 0.806,
  "screen_normalized_error": 0.0074,
  "timeout_ms": 1000
}
```

---

## 11. Tracking Task

### 11.1 Purpose

The tracking task measures how well the user can continuously follow a moving target.

### 11.2 Behavior

One circle moves around the screen using a seeded random path. The user must keep the cursor as close as possible to the circle center.

The app samples the cursor position at a fixed sampling rate.

### 11.3 Target Movement

The target should move smoothly, not jump randomly every tick.

Recommended MVP behavior:

- Generate seeded waypoints.
- Move smoothly from waypoint to waypoint.
- Keep the target inside safe screen margins.

### 11.4 Sampling Rate

Recommended MVP default:

```text
30 Hz
```

This means the app records cursor error about 30 times per second.

### 11.5 Duration

Recommended MVP default:

```text
30 seconds
```

### 11.6 Error Metrics

For each sample, calculate:

```text
pixel_error = distance between cursor and target center
radius_normalized_error = pixel_error / target_radius
screen_normalized_error = pixel_error / screen_diagonal
```

### 11.7 Final Outputs

The tracking task should output:

- Average pixel error
- Median pixel error
- Average radius-normalized error
- Median radius-normalized error
- Maximum error
- Stability score
- Tracking score

### 11.8 Raw Data Per Sample

Example:

```json
{
  "task": "tracking",
  "sample_index": 233,
  "timestamp_ms": 7766,
  "target_x": 700,
  "target_y": 460,
  "target_radius": 30,
  "cursor_x": 724,
  "cursor_y": 451,
  "pixel_error": 25.63,
  "radius_normalized_error": 0.854,
  "screen_normalized_error": 0.0118
}
```

---

## 12. Clicking Task

### 12.1 Purpose

The clicking task measures whether the user can click the correct target using the correct click type.

This task is especially important for testing eye-gesture clicking.

### 12.2 Behavior

A circle appears with an instruction:

```text
Left Click
```

or

```text
Right Click
```

The user must move the cursor to the circle and perform the requested click.

### 12.3 Success and Failure Rules

A click is a success when:

- The click is inside the target circle.
- The click type matches the requested click type.

A fail inside occurs when:

- The click is inside the target circle.
- The click type is wrong.

Example:

```text
Target asks for left click.
User right-clicks inside the circle.
Result: fail inside.
```

A fail outside occurs when:

- The user clicks outside the target circle.

Outside clicks should be recorded even if the click type is correct.

### 12.4 Timeout

Recommended MVP default:

```text
5 seconds per target
```

### 12.5 Final Outputs

The clicking task should output:

- Count trials
- Count success
- Count fails inside
- Count fails outside
- Timeout count
- Success rate
- Wrong-click rate
- Outside-click rate
- Clicking score

### 12.6 Raw Data Per Trial

Example:

```json
{
  "task": "clicking",
  "trial_index": 11,
  "target_x": 640,
  "target_y": 370,
  "target_radius": 35,
  "requested_click": "left",
  "actual_click": "right",
  "click_x": 651,
  "click_y": 362,
  "inside_target": true,
  "result": "fail_inside",
  "time_to_click_ms": 1840
}
```

---

## 13. Scoring System

The app should generate a final score from 0 to 100.

The final score should be easy to understand for demos, but raw metrics should remain available for serious analysis.

### 13.1 Final Score Weights

Recommended MVP weights:

```text
Final Score =
  30% Movement Score
+ 30% Accuracy Score
+ 25% Tracking Score
+ 15% Clicking Score
```

Reasons:

- Movement and accuracy are the most important cursor-control abilities.
- Tracking is important but slightly less central to ordinary desktop use.
- Clicking is important, but failures may depend on gesture design as much as cursor movement.

### 13.2 Score Labels

The final score should map to a quality label:

```text
90-100: Excellent
75-89:  Good
60-74:  Acceptable
40-59:  Poor
0-39:   Failed
```

### 13.3 Movement Score

Movement score should reward:

- High completion rate
- Low average movement time
- Low timeout rate

Recommended MVP formula:

```text
completion_component = completion_rate * 70
speed_component = clamp(1 - average_movement_time_ms / 3000, 0, 1) * 30
movement_score = completion_component + speed_component
```

### 13.4 Accuracy Score

Accuracy score should reward low normalized error.

Recommended MVP formula:

```text
accuracy_score = clamp(1 - average_radius_normalized_error / 3, 0, 1) * 100
```

This means:

- Error near 0 is excellent.
- Error around 1 radius is still decent.
- Error around 3 radii or more scores close to 0.

### 13.5 Tracking Score

Tracking score should reward low average normalized error and stability.

Recommended MVP formula:

```text
tracking_error_component = clamp(1 - average_radius_normalized_error / 4, 0, 1) * 80
stability_component = clamp(1 - error_std_dev / 3, 0, 1) * 20
tracking_score = tracking_error_component + stability_component
```

### 13.6 Clicking Score

Clicking score should reward successful clicks and penalize wrong or outside clicks.

Recommended MVP formula:

```text
success_component = success_rate * 100
inside_fail_penalty = fail_inside_rate * 35
outside_fail_penalty = fail_outside_rate * 50
timeout_penalty = timeout_rate * 25
clicking_score = clamp(success_component - inside_fail_penalty - outside_fail_penalty - timeout_penalty, 0, 100)
```

Outside clicks are penalized more heavily than wrong inside clicks because they indicate both a click event and a targeting failure.

---

## 14. Result Reports

### 14.1 Final Results Screen

At the end of a session, the app should show:

```text
Final Score: 82 / 100
Rating: Good

Movement: 86
Accuracy: 78
Tracking: 80
Clicking: 90
```

The screen should also show:

- Participant name
- Input method
- Seed
- Screen size
- Session date/time
- Task completion status
- Export buttons

### 14.2 Task Detail Screens

Each task should have a detailed result page.

Example movement details:

- Completed targets
- Timeouts
- Average movement time
- Fastest target
- Slowest target
- Movement score

Example accuracy details:

- Average pixel error
- Median pixel error
- Average normalized error
- Best trial
- Worst trial
- Accuracy score

---

## 15. Data Storage

The app should save data locally.

Recommended folder structure:

```text
EyeCursor TestLab/
  app_settings.json
  sessions/
    session_2026-04-25_14-30-10/
      session.json
      summary.json
      movement_trials.csv
      accuracy_trials.csv
      tracking_samples.csv
      clicking_trials.csv
      raw_events.json
  exports/
  logs/
```

### 15.1 Session JSON

Example:

```json
{
  "session_id": "session_2026-04-25_14-30-10",
  "participant_id": "participant_001",
  "participant_name": "User 1",
  "input_method": "Two-Camera Head Pose",
  "seed": 12345,
  "screen_width": 1920,
  "screen_height": 1080,
  "screen_diagonal_px": 2202.91,
  "started_at": "2026-04-25T14:30:10",
  "completed_at": null,
  "status": "in_progress",
  "completed_tasks": ["movement", "accuracy"],
  "next_task": "tracking"
}
```

### 15.2 Summary JSON

Example:

```json
{
  "session_id": "session_2026-04-25_14-30-10",
  "movement_score": 86.0,
  "accuracy_score": 78.0,
  "tracking_score": 80.0,
  "clicking_score": 90.0,
  "final_score": 82.3,
  "quality_label": "Good"
}
```

---

## 16. UI/UX Requirements

The UI should feel visually related to the main EyeCursor app.

It should use:

- Similar colors
- Similar typography
- Similar card style
- Similar button shapes
- Similar spacing
- Similar page structure

### 16.1 Main Layout

Use a modern desktop layout outside active tests:

```text
Left sidebar:
  Dashboard
  New Session
  Resume Session
  Results
  Settings

Main content:
  Cards
  Session setup
  Task progress
  Result summaries
```

### 16.2 Dashboard

Dashboard should show:

- Start new session button
- Resume session button
- Recent sessions
- Last final score
- Last tested input method
- Quick export button

### 16.3 Session Setup Page

The session setup page should include:

- Participant name or ID
- Input method
- Seed
- Task preset
- Notes
- Start fullscreen test button

### 16.4 Active Test Screen

The active test screen should be minimal.

It should show:

- Target circle
- Optional task instruction
- Optional progress indicator
- Optional timer

It should avoid sidebars, cards, and extra UI during the actual test because those can distract the user and affect results.

### 16.5 Transition Screen

Between tasks, show:

- Completed task name
- Quick task summary
- Continue button
- Pause session button
- End session button

### 16.6 Results Screen

The results screen should use cards:

```text
Final Score Card
Movement Card
Accuracy Card
Tracking Card
Clicking Card
Export Card
```

Each card should include a score, label, and key metrics.

---

## 17. Technical Architecture

The app should separate UI, task logic, scoring, and storage.

Recommended structure:

```text
src/
  app/
    main.py
    bootstrap.py
  ui/
    main_window.py
    pages/
      dashboard_page.py
      session_setup_page.py
      resume_session_page.py
      results_page.py
      settings_page.py
    test_window.py
    components/
      score_card.py
      task_card.py
      session_card.py
  core/
    sessions/
      session_manager.py
      session_model.py
    tasks/
      base_task.py
      movement_task.py
      accuracy_task.py
      tracking_task.py
      clicking_task.py
    scoring/
      scoring_engine.py
      score_models.py
    storage/
      storage_manager.py
      export_manager.py
    randomization/
      seeded_generator.py
    metrics/
      error_metrics.py
  resources/
    styles/
      app.qss
```

---

## 18. Task Interface

Every task should implement the same general interface.

Example:

```python
class TestTask:
    id: str
    display_name: str
    description: str

    def configure(self, config, seed):
        ...

    def start(self):
        ...

    def pause(self):
        ...

    def resume(self):
        ...

    def stop(self):
        ...

    def is_complete(self) -> bool:
        ...

    def get_raw_data(self) -> dict:
        ...

    def get_summary(self) -> dict:
        ...
```

Adding a future task should require:

1. Creating a new task class.
2. Registering it in the task registry.
3. Adding its scoring logic.
4. Adding its result card.

The rest of the app should not need major rewrites.

---

## 19. Cross-Platform Requirements

### 19.1 Priority

1. Windows
2. Linux
3. macOS

### 19.2 Windows

Requirements:

- Fullscreen mode works correctly.
- Cursor position sampling works correctly.
- Left and right click detection works correctly.
- PyInstaller build can launch successfully.

### 19.3 Linux

Requirements:

- Fullscreen mode works correctly.
- Cursor sampling works under the expected display server.
- If Wayland causes limitations, document them clearly.

### 19.4 macOS

Requirements:

- Fullscreen mode works correctly.
- Mouse and click events are detected.
- Any permission issues should be explained clearly.

---

## 20. Error Handling

The app should show friendly errors for:

- Failed to enter fullscreen
- Failed to read cursor position
- Invalid seed
- Incomplete session data
- Failed export
- Missing session file
- Corrupted result file
- Unsupported OS behavior

Bad:

```text
RuntimeError: failed
```

Good:

```text
Could not save the session results. Check that the app has permission to write to the results folder.
```

---

## 21. Development Roadmap

### Phase 1: App Shell and Style

- Create PySide6 app shell
- Add app theme and QSS styling
- Match the main EyeCursor look and feel
- Add sidebar navigation
- Add dashboard page
- Add basic settings page

Deliverable:

- App launches and looks visually consistent with EyeCursor.

### Phase 2: Session System

- Add session setup page
- Add participant and input method fields
- Add seed handling
- Add local session storage
- Add resume session support

Deliverable:

- User can create, save, stop, and resume a test session.

### Phase 3: Fullscreen Test Window

- Add fullscreen test runner
- Add keyboard safety controls
- Add transition screens
- Add task progress state

Deliverable:

- App can run a fullscreen task and return to the main UI.

### Phase 4: Movement and Accuracy Tasks

- Implement movement task
- Implement 100 ms dwell rule
- Implement accuracy task
- Save raw trial data
- Calculate task summaries

Deliverable:

- User can complete movement and accuracy tests.

### Phase 5: Tracking Task

- Implement moving target path
- Add 30 Hz sampling
- Save tracking samples
- Calculate tracking metrics

Deliverable:

- User can complete the tracking test.

### Phase 6: Clicking Task

- Implement left/right click targets
- Record correct clicks
- Record wrong inside clicks
- Record outside clicks
- Calculate clicking metrics

Deliverable:

- User can complete the clicking test.

### Phase 7: Scoring and Results

- Implement scoring engine
- Add final 0-100 score
- Add score labels
- Add result cards
- Add JSON and CSV export

Deliverable:

- User can view and export full test results.

### Phase 8: Packaging and Testing

- Package Windows build
- Test on Windows
- Package Linux build
- Test on Linux
- Package macOS build if possible
- Document known limitations

Deliverable:

- Working build ready for university project testing.

---

## 22. MVP Acceptance Criteria

The MVP is complete when:

1. The app launches as a PySide6 desktop app.
2. The UI visually matches the main EyeCursor app style.
3. The user can start a new test session.
4. The user can select an input method.
5. The user can enter or generate a seed.
6. The app runs tests in fullscreen mode.
7. The movement task works with 100 ms dwell time.
8. The accuracy task records pixel and normalized error.
9. The tracking task records sampled cursor error.
10. The clicking task records success, fail inside, fail outside, and timeouts.
11. The user can complete all tests sequentially.
12. The user can stop and continue a session.
13. Session data is saved after every completed task.
14. Raw data is stored in JSON.
15. Summary data can be exported to CSV.
16. The app calculates task scores.
17. The app calculates a final 0-100 score.
18. The app shows a final quality label.
19. `Esc` pauses or exits the active test.
20. `Q` stops the active test.
21. The app can be packaged with PyInstaller.

---

## 23. Final MVP Recommendation

Build EyeCursor TestLab as a polished PySide6 fullscreen desktop application that runs independent cursor-control tests.

The MVP should focus on:

- Repeatable seeded tests
- Fullscreen behavior
- Movement speed
- Positional accuracy
- Continuous tracking accuracy
- Correct left/right click detection
- Raw data logging
- A clear final score

The most important technical rule is that the app should save raw data, not just averages.

The most important UX rule is that the test should feel simple during execution and polished during setup/results.

The most important research rule is that the same seed and task settings must produce the same targets so that different cursor-control methods can be compared fairly.
