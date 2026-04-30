"""Default thresholds and timings for the blendshape-driven GestureController.

Per-profile calibration overrides any of these via the controller's
constructor; the values here are used when no calibration is loaded
or when a calibration field is missing.
"""

# Smirk -> click (and click-and-hold for drawing)
SMIRK_TRIGGER_DIFF = 0.40   # |left_activation - right_activation| above this fires a click
SMIRK_RELAX_DIFF = 0.15     # |left_activation - right_activation| must drop below this to re-arm

# Click-and-hold: while a smirk is held, cursor stays frozen for this long
# (so the press lands on a stable target). After this elapses the cursor
# starts moving again WHILE the button stays held -- this is what enables
# drawing/dragging.
CLICK_HOLD_UNFREEZE_SEC = 0.5

# Cheek puff -> scroll DOWN
CHEEK_PUFF_RELEASE = 0.25       # below this: fully idle, intent timer resets
CHEEK_PUFF_DOWN_LOW = 0.30      # entering [low, high] starts the scroll-down intent buffer
CHEEK_PUFF_DOWN_HIGH = 0.70     # at or above this: max-speed scroll down (saturates above)
CHEEK_PUFF_UP_HIGH = 1.00       # legacy field kept for v3 calibration compat; unused in v4

# Lip tuck-in -> scroll UP. Default source is
# max(mouthRollUpper, mouthRollLower, mouthPressLeft, mouthPressRight).
# These blendshapes co-activate when the user rolls/tucks their lips inward
# or presses them firmly together; they're MediaPipe's most reliable
# "lips-pulled-in" signal.
TUCK_RELEASE = 0.15
TUCK_TRIGGER_LOW = 0.30
TUCK_TRIGGER_HIGH = 0.70

# Anti-jitter: 200 ms intent buffer for both scroll directions.
SCROLL_INTENT_DELAY_SEC = 0.20

# Speed mapping for variable-rate scroll ticks. Tuned down from the original
# 60-600 / 120-tick defaults because that combination scrolled an entire
# screen in well under a second on typical apps -- way too fast.
SCROLL_MIN_UNITS_PER_SEC = 20.0
SCROLL_MAX_UNITS_PER_SEC = 200.0
SCROLL_TICK_DELTA = 30
SCROLL_MIN_TICK_INTERVAL_SEC = 0.05
