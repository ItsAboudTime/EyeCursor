"""Stable camera identification across reboots and USB replugs.

Cameras are normally addressed by their numeric `/dev/videoN` index, but Linux
USB enumeration is non-deterministic: the same physical camera may land at a
different index across reboots, or after the user replugs a USB cable. This
module derives an identifier that is stable across those events as long as the
same physical camera is plugged into the same USB port (and, when the device's
USB serial number is unique, it is stable across port changes too).

Identifier format
-----------------
The identifier is a colon-separated string with one of two shapes:

* ``usb:{vendor}:{product}:serial:{serial}`` — used when the device exposes a
  serial number that is unlikely to collide with another unit of the same
  model (i.e. it is not empty, not equal to ``vendor:product``, and not equal
  to the device's product/model name -- some webcams hard-code "WebCamera" or
  similar as the "serial").
* ``usb:{vendor}:{product}:port:{usb_port}`` — used as a fallback when the
  device's serial cannot be trusted. ``usb_port`` is the bus path the kernel
  assigns based on the physical USB topology (e.g. ``3-3`` or ``3-2.4``);
  it is stable as long as the cable stays in the same port.
* ``index:{n}`` — last-resort fallback when neither of the above can be
  determined (e.g. on non-Linux platforms where this module returns ``None``,
  or when the sysfs entries are missing). Calibration code is expected to
  treat ``index:*`` IDs as opaque and to bail out (with a clear error) when
  they don't match. Since ``index:*`` IDs are not stable, they should be
  treated as "no stable ID known".

The functions in this module return ``None`` when no stable ID can be derived
so callers can decide whether to use ``index:{n}`` or to leave the field
absent entirely.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Optional


__all__ = [
    "stable_id_for_index",
    "build_stable_id",
    "extract_index_from_stable_id",
    "INDEX_PREFIX",
    "USB_PREFIX",
]


INDEX_PREFIX = "index:"
USB_PREFIX = "usb:"

# Common placeholder serials that are not actually unique. UVC webcams from
# generic suppliers often hard-code their model name (or even an empty string)
# as the serial. Treat these as "no usable serial" and fall back to the USB
# port path.
_BAD_SERIAL_TOKENS = {
    "",
    "0",
    "0000",
    "00000000",
    "n/a",
    "na",
    "none",
    "null",
    "default",
    "unknown",
    "(none)",
    # Built-in webcams often expose a firmware revision rather than a
    # per-unit serial. These are stable per-model but not per-unit and
    # would collide between two physical cameras of the same model.
    "01.00.00",
    "00.00.00",
    "0.0.0",
    "1.0.0",
}


def _read_sysfs(path: Path) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except (OSError, ValueError):
        return None


def _looks_like_useful_serial(
    serial: Optional[str],
    vendor: Optional[str],
    product: Optional[str],
    model: Optional[str],
) -> bool:
    """Reject serials that are clearly not unique per physical unit."""
    if not serial:
        return False
    s = serial.strip().lower()
    if s in _BAD_SERIAL_TOKENS:
        return False
    # Some cameras report `vendor:product` as the serial.
    if vendor and product and s == f"{vendor.lower()}:{product.lower()}":
        return False
    # Some cameras report the model/product name as the serial -- not unique
    # across physical units.
    if model and s == model.strip().lower():
        return False
    if product and s == product.strip().lower():
        return False
    # Trim leading/trailing zeros and check again -- "0000" sometimes shows
    # up after stripping.
    if all(c == "0" for c in s):
        return False
    return True


def _usb_port_for_video(video_node: str) -> Optional[str]:
    """Return the kernel USB port path (e.g. ``3-3``) for a /dev/videoN node.

    Linux exposes the parent USB device of a v4l2 node via
    /sys/class/video4linux/videoN/device -> ../<usb_iface>. The grandparent
    of that is the USB device itself, whose directory name is the bus path.
    """
    sys_dev = Path(f"/sys/class/video4linux/{video_node}/device")
    if not sys_dev.exists():
        return None
    try:
        # Resolve to the interface directory and walk up one level to the
        # USB device directory. The basename of that path is the bus path.
        iface = sys_dev.resolve()
        usb_dev_dir = iface.parent
        return usb_dev_dir.name or None
    except OSError:
        return None


def build_stable_id(
    vendor_id: Optional[str],
    product_id: Optional[str],
    serial: Optional[str],
    usb_port: Optional[str],
    product_name: Optional[str] = None,
) -> Optional[str]:
    """Compose a stable identifier from raw device metadata.

    Returns ``None`` when not enough information is available. When the
    serial looks unique it is preferred (so the camera follows the user
    even when they move it to a different USB port); otherwise the USB
    port path is used.
    """
    vendor = (vendor_id or "").strip().lower()
    product = (product_id or "").strip().lower()
    if not vendor or not product:
        return None

    if _looks_like_useful_serial(serial, vendor, product, product_name):
        return f"{USB_PREFIX}{vendor}:{product}:serial:{serial.strip()}"

    if usb_port:
        return f"{USB_PREFIX}{vendor}:{product}:port:{usb_port.strip()}"

    return None


def stable_id_for_index(index: int) -> Optional[str]:
    """Best-effort stable identifier for a `/dev/videoN` index.

    On non-Linux platforms returns ``None`` (cameras still work via index;
    the migration path will simply leave stable IDs absent).
    """
    if platform.system() != "Linux":
        return None
    video_node = f"video{int(index)}"
    sys_dev = Path(f"/sys/class/video4linux/{video_node}/device")
    if not sys_dev.exists():
        return None

    # The device symlink points at the USB interface; the USB device itself
    # is one directory up.
    try:
        usb_dev_dir = sys_dev.resolve().parent
    except OSError:
        return None

    vendor = _read_sysfs(usb_dev_dir / "idVendor")
    product = _read_sysfs(usb_dev_dir / "idProduct")
    serial = _read_sysfs(usb_dev_dir / "serial")
    product_name = _read_sysfs(usb_dev_dir / "product")
    usb_port = _usb_port_for_video(video_node)

    return build_stable_id(
        vendor_id=vendor,
        product_id=product,
        serial=serial,
        usb_port=usb_port,
        product_name=product_name,
    )


def extract_index_from_stable_id(stable_id: Optional[str]) -> Optional[int]:
    """If the stable ID is the `index:N` last-resort form, return N."""
    if not stable_id:
        return None
    if stable_id.startswith(INDEX_PREFIX):
        try:
            return int(stable_id[len(INDEX_PREFIX):])
        except ValueError:
            return None
    return None
