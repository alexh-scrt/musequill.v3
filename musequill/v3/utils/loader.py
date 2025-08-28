import json
from pathlib import Path

def load_chapter_briefs(folder: str, start: int = 1, end: int = 20) -> dict[int, dict]:
    """
    Load chapter briefs from JSON files in a folder.

    Expected filenames: chapter-<n>-brief.json where n = 1..end.

    Args:
        folder: Path to folder containing JSON files.
        start: First chapter number (default=1).
        end: Last chapter number (default=20).

    Returns:
        Dictionary mapping chapter number -> loaded JSON content.
    """
    folder_path = Path(folder)
    chapters = {}

    for i in range(start, end + 1):
        file_path = folder_path / f"chapter-{i}-brief.json"
        if not file_path.exists():
            print(f"⚠️ Missing file: {file_path}")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                chapters[i] = json.load(f)
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing {file_path}: {e}")

    return chapters
