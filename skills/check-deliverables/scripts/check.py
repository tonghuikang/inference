#!/usr/bin/env python3
"""Block Stop until the assistant confirms it addressed every user requirement.

Reads a Claude Code transcript (JSONL), inspects the latest assistant turn,
and:
  - exits 0 if the latest assistant text contains the confirmation phrase
  - exits 2 with a checklist on stderr otherwise (Stop hook then blocks)

The check runs on every Stop regardless of whether files were edited.
"""

import json
import sys
from pathlib import Path

CONFIRMATION_PHRASE = "I have addressed every query from the user."

CHECKLIST = f"""Run the check-deliverables review.

Do not rationalize. Look for evidence the work was done, not arguments that it
was.

1. Enumerate every requirement from the user
   - Quote the user's instruction verbatim
   - Add to TaskCreate

2. For each requirement, present evidence (not reasoning):
   - A file path + line range you changed, or
   - A command you ran with its output, or
   - A tool call you made
   If no concrete artifact exists, mark UNADDRESSED and finish the work
   before stopping.

3. Only when every requirement has concrete evidence, end your response with:
{CONFIRMATION_PHRASE}
"""


def _latest_assistant_entry(transcript_path: str) -> dict | None:
    path = Path(transcript_path)
    if not path.exists():
        return None
    for line in reversed(path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        if entry.get("type") == "assistant":
            return entry
    return None


def check(transcript_path: str) -> list[str]:
    entry = _latest_assistant_entry(transcript_path)
    if entry is None:
        return []

    contents = entry.get("message", {}).get("content", [])
    text = "\n".join(c.get("text", "") for c in contents if c.get("type") == "text")
    if CONFIRMATION_PHRASE in text:
        return []

    return [
        f"Confirmation phrase is missing. End your response with: '{CONFIRMATION_PHRASE}'",
        CHECKLIST,
    ]


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: check.py <transcript_path>", file=sys.stderr)
        return 1
    issues = check(sys.argv[1])
    for issue in issues:
        print(issue, file=sys.stderr)
    return 2 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
