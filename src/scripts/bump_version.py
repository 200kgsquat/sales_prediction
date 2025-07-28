# bump_version.py
import tomlkit
from pathlib import Path

pyproject_path = Path("pyproject.toml")
pyproject_text = pyproject_path.read_text()
doc = tomlkit.parse(pyproject_text)

version_str = doc["project"]["version"]
major, minor, patch = map(int, version_str.split("."))

patch += 1
new_version = f"{major}.{minor}.{patch}"
doc["project"]["version"] = new_version

pyproject_path.write_text(tomlkit.dumps(doc))
print(f"Version bumped to: {new_version}")
