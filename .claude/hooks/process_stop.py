"""
Claude Code Hook: Stop Validator.

Delegates to the check-deliverables skill script. The script inspects the
latest assistant turn and asks the assistant to confirm it addressed every
user requirement. A non-empty stderr from the script becomes an exit-2
message, which blocks Stop and feeds the message back to Claude.
"""

import os
import subprocess
import sys
from pathlib import Path

CHECK_SCRIPT = (
    Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))
    / "skills"
    / "check-deliverables"
    / "scripts"
    / "check.py"
)


def validate_stop(transcript_path: str) -> list[str]:
    """Run the check-deliverables script; return its stderr lines if it failed."""
    if not CHECK_SCRIPT.exists() or not transcript_path:
        return []

    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), transcript_path],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return []

    message = result.stderr.strip()
    return [message] if message else []
