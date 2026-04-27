import json
import os
import tempfile
from pathlib import Path

from process_stop import validate_stop


def _write_transcript(entries: list[dict]) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for entry in entries:
        f.write(json.dumps(entry) + "\n")
    f.close()
    return f.name


def _set_project_dir() -> None:
    os.environ["CLAUDE_PROJECT_DIR"] = str(Path(__file__).resolve().parents[2])


def test_stop_validator_no_edits_no_phrase_blocks():
    """Even without edits, missing the confirmation phrase blocks Stop."""
    _set_project_dir()
    transcript_path = _write_transcript(
        [
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Some response without edits"}]
                },
            }
        ]
    )
    try:
        issues = validate_stop(transcript_path)
        assert len(issues) == 1
        assert "Confirmation phrase is missing" in issues[0]
    finally:
        Path(transcript_path).unlink()


def test_stop_validator_no_edits_with_phrase():
    """No edits but phrase present → no issues."""
    _set_project_dir()
    transcript_path = _write_transcript(
        [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": "Done. I have addressed every query from the user.",
                        }
                    ]
                },
            }
        ]
    )
    try:
        assert validate_stop(transcript_path) == []
    finally:
        Path(transcript_path).unlink()


def test_stop_validator_with_edits_and_confirmation():
    """Edits + confirmation phrase → no issues."""
    _set_project_dir()
    transcript_path = _write_transcript(
        [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Edit",
                            "input": {
                                "old_string": "foo",
                                "new_string": "bar",
                                "file_path": "x.py",
                            },
                        },
                        {
                            "type": "text",
                            "text": "I have addressed every query from the user.",
                        },
                    ]
                },
            }
        ]
    )
    try:
        assert validate_stop(transcript_path) == []
    finally:
        Path(transcript_path).unlink()


def test_stop_validator_with_edits_no_confirmation():
    """Edits without confirmation phrase → block Stop with a checklist message."""
    _set_project_dir()
    transcript_path = _write_transcript(
        [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Edit",
                            "input": {
                                "old_string": "foo",
                                "new_string": "bar",
                                "file_path": "x.py",
                            },
                        },
                        {"type": "text", "text": "Some other text"},
                    ]
                },
            }
        ]
    )
    try:
        issues = validate_stop(transcript_path)
        assert len(issues) == 1
        assert "Confirmation phrase is missing" in issues[0]
        assert "I have addressed every query from the user." in issues[0]
    finally:
        Path(transcript_path).unlink()


def test_stop_validator_missing_transcript_path():
    """Empty transcript path → no issues (defensive default)."""
    _set_project_dir()
    assert validate_stop("") == []
