from typing import Optional


def derive_last_action(
    last_click_side: Optional[str],
    pre_scroll: Optional[str],
    post_scroll: Optional[str],
) -> Optional[str]:
    """Surface the most recent gesture action for the visualizer.

    Returns one of: 'left_click_down', 'right_click_down',
    'scroll_up', 'scroll_down', or None.

    ``last_click_side`` is the controller's one-frame click indicator
    (``GestureController._last_click_side``); when set, a click just fired.
    Scroll transitions are reported once per state change.
    """
    if last_click_side == "left":
        return "left_click_down"
    if last_click_side == "right":
        return "right_click_down"
    if pre_scroll != post_scroll and post_scroll is not None:
        if post_scroll == "scroll_up":
            return "scroll_up"
        if post_scroll == "scroll_down":
            return "scroll_down"
    return None
