---
name: check-deliverables
description: Review work against user requirements, enumerate tasks, and verify completion with tests and formatting
---

Review your work.

Do not rationalize. Justifications you produce after deciding you're done are
not verification — they're defense. Use this skill to look for evidence the
work was actually done, not to argue that it was.

You will

1) Enumerate every requirement from the user
    - Quote the user's instruction verbatim
    - Add to the `TaskCreate` tool

2) For each requirement, present **evidence**, not reasoning
    - A file path and line range you changed, or a command you ran with its
      output, or a tool call you made
    - If you have no concrete artifact to point to, mark the requirement
      UNADDRESSED and stop — do not invent justification

3) If any requirement is UNADDRESSED, finish the work before stopping. Do not
   end the response with the confirmation phrase.

When every requirement has concrete evidence, end your response with the exact
phrase:

    I have addressed every query from the user.

The Stop hook (`.claude/hooks/process_stop.py`) runs `scripts/check.py` from this
skill on every Stop. If the latest assistant turn does not include the phrase
above, the hook exits 2, blocking Stop and asking you to complete the review.
The check runs regardless of whether any files were edited.
