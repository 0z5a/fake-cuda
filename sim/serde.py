"""External profile artifacts; raw calibration/session files are never git inputs."""
import json
from pathlib import Path
from .cost import Identity, Profile


def read_profiles(path: Path) -> tuple[Identity, tuple[Profile, ...]]:
    document = json.loads(path.read_text())
    if document["schema"] != 1:
        raise ValueError("unsupported calibration artifact schema")
    identity = Identity(**document["identity"])
    profiles = []
    for original in document["profiles"]:
        entry = dict(original)
        entry["identity"] = Identity(**entry["identity"])
        entry["family"] = tuple(entry["family"])
        entry["samples"] = tuple((tuple(features), ns) for features, ns in entry["samples"])
        for key in ("coefficients", "lower", "upper"):
            entry[key] = tuple(entry[key])
        profiles.append(Profile(**entry))
    return identity, tuple(profiles)
