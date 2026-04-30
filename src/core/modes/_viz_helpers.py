from typing import Optional


def derive_last_action(
    pre_blink: Optional[str],
    post_blink: Optional[str],
    pre_scroll: Optional[str],
    post_scroll: Optional[str],
) -> Optional[str]:
    """Diff gesture-controller state across one frame to surface the most recent action.

    Returns one of: 'left_click_down', 'left_click_up', 'right_click_down',
    'right_click_up', 'scroll_up', 'scroll_down', or None.
    """
    if pre_blink != post_blink:
        side = post_blink if post_blink else pre_blink
        if side == "left":
            return "left_click_down" if post_blink == "left" else "left_click_up"
        if side == "right":
            return "right_click_down" if post_blink == "right" else "right_click_up"
    if pre_scroll != post_scroll and post_scroll is not None:
        if post_scroll == "both_open":
            return "scroll_up"
        if post_scroll == "both_squint":
            return "scroll_down"
    return None
