from typing import Dict, Iterable, Optional, Tuple


BLENDSHAPE_KEYS = (
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthPucker",
    "mouthRollUpper",
    "mouthRollLower",
    "mouthPressLeft",
    "mouthPressRight",
)


def extract_blendshapes(categories: Optional[Iterable]) -> Dict[str, float]:
    """Pull just the blendshape scores we care about into a plain dict.

    Accepts the ``face_blendshapes[0]`` list returned by MediaPipe (each entry
    has ``category_name`` and ``score`` attributes), or ``None`` when the
    landmarker did not emit blendshapes for this frame. Missing keys default
    to 0.0 so downstream consumers never need to special-case absence.
    """
    result = {key: 0.0 for key in BLENDSHAPE_KEYS}
    if categories is None:
        return result

    for category in categories:
        name = getattr(category, "category_name", None)
        if name in result:
            score = getattr(category, "score", 0.0)
            try:
                result[name] = float(score)
            except (TypeError, ValueError):
                result[name] = 0.0
    return result


def compute_smirk_activations(blendshapes: Dict[str, float]) -> Tuple[float, float]:
    """Return (left_activation, right_activation) for smirk detection.

    Uses ``mouthSmileLeft`` / ``mouthSmileRight`` only.
    """
    left = float(blendshapes.get("mouthSmileLeft", 0.0))
    right = float(blendshapes.get("mouthSmileRight", 0.0))
    return left, right


def pucker_value(blendshapes: Dict[str, float]) -> float:
    """Return mouth-pucker intensity (used to fire LEFT click).

    Reads MediaPipe's ``mouthPucker`` blendshape directly. We intentionally
    do NOT fall back to ``cheekPuff`` -- in MediaPipe's default model that
    score is unreliable and adds noise; ``mouthPucker`` (lips pushed
    outward, as if blowing/kissing) is the stable signal.
    """
    return float(blendshapes.get("mouthPucker", 0.0))


def tuck_value(blendshapes: Dict[str, float]) -> float:
    """Return lip-tuck intensity (used to fire RIGHT click).

    Activates when the user tucks/rolls their lips inward or presses them
    firmly together. Uses the max of four MediaPipe blendshapes that all
    co-activate for this gesture: ``mouthRollUpper`` and ``mouthRollLower``
    (literal "rolling lips inward"), plus ``mouthPressLeft`` and
    ``mouthPressRight`` (lips pressed together). Taking the max makes the
    detector robust to per-user variation in which blendshape MediaPipe
    scores most strongly.
    """
    return max(
        float(blendshapes.get("mouthRollUpper", 0.0)),
        float(blendshapes.get("mouthRollLower", 0.0)),
        float(blendshapes.get("mouthPressLeft", 0.0)),
        float(blendshapes.get("mouthPressRight", 0.0)),
    )
