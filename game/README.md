# Horsin' Around

A standalone Panda3D demo built to showcase the EyeCursor head/eye-gaze cursor-control system. The player rides an automated rollercoaster around an oval track and photographs procedurally placed horses by holding a configurable trigger; during the countdown, leaning toward the screen telephoto-zooms the lens via stereo depth piped over a UDP socket. QUAKE-style pixel aesthetic.

## Demo

> Screenshot coming soon.

## Stack

| Concern    | Choice                                  |
|------------|-----------------------------------------|
| Engine     | Panda3D >= 1.10.14                      |
| Language   | Python 3.12                             |
| Aesthetic  | QUAKE / pixel-art (nearest-neighbor)    |
| IPC        | UDP localhost:7345 for stereo depth     |

## Quick start

The game is intended to run from the parent EyeCursor project's `venv`.

```bash
./venv/bin/pip install -r requirements.txt   # installs panda3d
./launch_game.sh
```

Both commands are run from the repository root.

## Controls

| Input                                                | Action                                                         |
|------------------------------------------------------|----------------------------------------------------------------|
| Mouse / OS-cursor                                    | Look around (cart-relative; centered = look down the track)    |
| Photo trigger (left click / spacebar / right click)  | Hold to compose and shoot                                      |
| Up / Down arrows                                     | Navigate menus                                                 |
| Left / Right arrows (in Settings)                    | Cycle the focused option backward / forward                    |
| Enter / Return                                       | Activate focused menu option                                   |
| ESC (in ride)                                        | Pause overlay                                                  |

The photo trigger is configurable in the Settings screen; the default is left click.

## Settings

Settings are persisted to `game/config.json` and written immediately on change.

| Key                  | Valid values                                | Notes                                                     |
|----------------------|---------------------------------------------|-----------------------------------------------------------|
| `photo_trigger`      | `left_click`, `spacebar`, `right_click`     | Which input arms the photo countdown.                     |
| `countdown_duration` | `0.5`, `1.0`, `1.5`, `2.0` (seconds)        | Time the trigger must be held before the shutter fires.   |
| `cart_speed`         | `slow`, `normal`, `fast`                    | Mapped to `1.5`, `3.0`, `6.0` units/sec - scenic by design. |

## How the depth zoom works

- When the user starts the EyeCursor "Two-Camera Head Pose" mode, a `DepthBroadcaster` (in `src/core/modes/two_camera_head_pose.py`) emits stereo-derived face depth as JSON over UDP `127.0.0.1:7345` at ~30 Hz.
- The game runs a daemon thread that reads those packets into a single `latest_depth` value (`game/core/depth_client.py`), with a 2-second freshness cutoff.
- When the photo trigger is held, the game captures a *baseline depth* and then maps the change in distance (using `abs()` so the stereo coordinate sign convention does not matter - closer = smaller `|depth|`) to FOV: leaning ~20 cm closer to the camera = full zoom (FOV -> 30 deg), no movement = no zoom (FOV stays at 90 deg).
- Pulling away from the baseline never zooms (clamped). The baseline resets after each photo.
- If the depth client receives no packets (stereo mode not running), FOV stays at 90 deg - the game still plays normally.

## Project layout

```
game/
├── app/         # Panda3D ShowBase entry point and scene manager
├── core/        # Track, camera, photo manager, depth client, settings, asset gen
├── scenes/      # Main menu, settings, game, pause overlay, gallery
├── assets/      # Generated textures and the downloaded pixel font
└── photos/      # Saved PNG screenshots (git-ignored)
```

## Assets

- **Horse model**: procedural - built from stacked Panda3D primitive boxes (no `.egg` file shipped).
- **Ground / sky textures**: procedurally generated 64x64 PNGs on first launch (`game/core/asset_gen.py`), loaded with nearest-neighbor filtering for the chunky pixel feel.
- **Pixel font**: Press Start 2P (OFL-licensed) auto-downloaded on first launch from the Google Fonts repo into `game/assets/fonts/pixel.ttf`. Falls back gracefully to Panda3D's default font if the download fails.

## Photo preview

After every capture, a thumbnail of the just-saved PNG slides up from the bottom-right corner of the screen (Marvel's Spider-Man-style), holds for ~2.5 s, then slides back out. Bordered in the QUAKE accent green so it reads against the world.

## Photos location

Saved screenshots land in `game/photos/YYYY-MM-DD_HH-MM-SS.png`. The directory is git-ignored.

## License

Game code is part of the EyeCursor graduation project and inherits the parent repository's license. Press Start 2P is OFL-1.1 from Google Fonts.
