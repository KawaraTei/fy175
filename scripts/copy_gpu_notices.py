"""Preserve installed NVIDIA wheel notices and exact versions in local GPU builds."""
from importlib.metadata import distributions
from pathlib import Path
import shutil
import sys


def main():
    destination = Path(sys.argv[1])
    versions = []
    for dist in distributions():
        name = dist.metadata["Name"]
        if not name.lower().startswith("nvidia-"):
            continue
        copied = 0
        for item in dist.files or ():
            if "licenses" not in item.parts:
                continue
            source = Path(dist.locate_file(item))
            if source.is_file():
                relative = Path(*item.parts[item.parts.index("licenses") + 1:])
                target = destination / "LICENSES" / f"{name}-{dist.version}" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied += 1
        if not copied:
            raise RuntimeError(f"No license files found for {name}")
        versions.append(f"{name}=={dist.version}")
    if not versions:
        raise RuntimeError("No NVIDIA runtime packages found")
    (destination / "GPU_RUNTIME.txt").write_text("\n".join(sorted(versions)) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
