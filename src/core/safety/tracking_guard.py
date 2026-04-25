class TrackingGuard:
    def __init__(self) -> None:
        self._is_tracking = False
        self._is_calibrating = False

    @property
    def is_tracking(self) -> bool:
        return self._is_tracking

    @property
    def is_calibrating(self) -> bool:
        return self._is_calibrating

    def can_start_tracking(self) -> tuple:
        if self._is_calibrating:
            return False, "Cannot start tracking during calibration."
        if self._is_tracking:
            return False, "Tracking is already active."
        return True, ""

    def can_start_calibration(self) -> tuple:
        if self._is_tracking:
            return False, "Cannot calibrate while tracking is active. Stop tracking first."
        return True, ""

    def start_tracking(self) -> None:
        self._is_tracking = True

    def stop_tracking(self) -> None:
        self._is_tracking = False

    def start_calibration(self) -> None:
        self._is_calibrating = True

    def stop_calibration(self) -> None:
        self._is_calibrating = False
